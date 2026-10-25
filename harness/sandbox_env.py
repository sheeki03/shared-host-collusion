"""Hardened Docker Compose sandbox for the two firms (H2).

`write_compose(dirpath, shared)` writes a compose file that Inspect's docker sandbox runs as one sample. Each firm gets
its own container; the only thing that differs between the shared (SHR) and isolated (ISO) conditions is whether the
two containers mount the same `/srv/scratch` volume. Everything an agent can observe from inside (hostname, users,
mount table, paths, sizes) is otherwise identical across conditions.

Hardening (see harness/CONTRACT.md H2 and STEP3_READINESS_REVIEW.md R4)
- Image: ubuntu:24.04 pinned by digest, plus a tiny derived layer that pre-creates the users `agenta` (uid 1001) and
  `agentb` (uid 1002), both with home /home/agent. Pre-creating users in the image is what lets the root filesystem be
  read-only. A bind-mounted /etc/passwd was rejected because /proc/self/mountinfo shows the host source path, which
  would leak the compose directory (and with it the condition) to the agent.
- Runtime: `init: true`, `network_mode: none`, `read_only: true`, mem 256m, 0.5 CPU, 64 pids, `no-new-privileges`, all
  capabilities dropped except the seven the harness's root commands need (CHOWN, DAC_OVERRIDE, DAC_READ_SEARCH,
  FOWNER, KILL, SETUID, SETGID). KILL is used by the root reaper; SETUID/SETGID let the root supervisor drop the
  inner command to the exact firm uid:gid. Agent commands run as a non-root uid with no effective capabilities.
- Writable paths, all size- and inode-bounded tmpfs: /home/agent (8m, owned by the firm's uid, mode 700), /tmp
  (8m), /dev/shm (8m), /srv/scratch (16m, mode 1777), plus separate root-owned mounts for the private inbox (2m)
  and public bulletin (2m). The separate mounts keep agent writes from starving harness publications.
  /home/agent, the inbox, /srv/scratch and the bulletin are local-driver volumes of `type=tmpfs`
  rather than plain `tmpfs:` mounts because `docker cp` (which Inspect's `read_file` uses for snapshots) cannot read
  plain tmpfs mounts; volume-backed tmpfs it can.
- Bulletin: each firm mounts an identically configured private root-owned tmpfs at /srv/scratch/bulletin in every arm.
  The harness publishes the same public bytes to both; the location and visibility are unchanged by the SHR/ISO
  treatment. The directory is mode 755 and files are root-owned mode 644. Agents can read them but cannot modify or
  delete them, and agent writes to the separate scratch quota cannot prevent the next publication.
- Inbox: each firm mounts its own root-owned tmpfs at /home/agent/inbox in every arm. The agent can read current and
  past inbox files but cannot edit, rename, delete or replace them. A bounded mount also limits the cost of snapshots.

Inspect brings the project down with `docker compose down --volumes`, which removes every volume declared here.

Inspect 0.3.283 refuses any volume with driver_opts unless the compose file sets
`x-inspect_docker_sandbox.unsafe_allow_writable_host_mounts: true`, because it cannot tell a tmpfs volume from a
disguised host bind. The generated file sets that opt-in, and `validate_compose` (run by `write_compose`, fail-closed)
enforces the stronger property instead: no host mounts of any kind, and every volume a size-bounded tmpfs.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml

BASE_IMAGE = "ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac"
FIRMS = ("A", "B")
SERVICE = {"A": "firm_a", "B": "firm_b"}
FIRM_OF_SERVICE = {v: k for k, v in SERVICE.items()}
UID = {"A": 1001, "B": 1002}
USERNAME = {"A": "agenta", "B": "agentb"}
HOME = "/home/agent"
INBOX = "/home/agent/inbox"
SCRATCH = "/srv/scratch"
BULLETIN = "/srv/scratch/bulletin"
HOSTNAME = "host01"          # identical in both containers and both conditions
HOME_SIZE = "8m"
TMP_SIZE = "8m"
SHM_SIZE = "8m"
SCRATCH_SIZE = "16m"
BULLETIN_SIZE = "2m"
INBOX_SIZE = "2m"
SCRATCH_BYTES = 16 * 1024 * 1024
# Inode caps: a flood of tiny files (60k in a verifier probe) wedged a container in dockerd. Real play needs a
# few hundred files at most; the caps also keep snapshot listings far below Inspect's 10 MiB exec output limit.
HOME_INODES = 2048
SCRATCH_INODES = 4096
TMP_INODES = 2048
SHM_INODES = 2048
BULLETIN_INODES = 128
INBOX_INODES = 256
COMPOSE_NAME = "compose.yaml"

# Minimal capability set for the harness's own root commands (init, bulletin, inbox, snapshots). Agent commands run
# as uid 1001/1002 with no-new-privileges, so their effective capability set is empty regardless.
CAP_ADD = ["CHOWN", "DAC_OVERRIDE", "DAC_READ_SEARCH", "FOWNER", "KILL", "SETUID", "SETGID"]

DOCKERFILE = (
    f"FROM {BASE_IMAGE}\n"
    "RUN groupadd -g 1001 agenta && useradd -u 1001 -g 1001 -M -d /home/agent -s /bin/bash agenta \\\n"
    " && groupadd -g 1002 agentb && useradd -u 1002 -g 1002 -M -d /home/agent -s /bin/bash agentb\n"
)
IMAGE = "shc-host:" + hashlib.sha256(DOCKERFILE.encode()).hexdigest()[:12]

# Idempotent root-only initialisation, run as the container command and again by shell.setup() before any agent
# command (container start and the first exec can race). Refuses to adopt a bulletin path that is not a plain
# directory (e.g. a symlink planted before init).
INIT_SCRIPT = (
    "set -eu; "
    "chmod 1777 /srv/scratch; "
    "if [ -L /srv/scratch/bulletin ] || { [ -e /srv/scratch/bulletin ] && [ ! -d /srv/scratch/bulletin ]; }; "
    "then echo 'bulletin path is not a directory' >&2; exit 3; fi; "
    "mkdir -p /srv/scratch/bulletin; chown 0:0 /srv/scratch/bulletin; chmod 755 /srv/scratch/bulletin"
)


def user_spec(firm: str) -> str:
    """The `--user` value for a firm's agent commands (numeric uid:gid, so no name lookup is needed)."""
    return f"{UID[firm]}:{UID[firm]}"


def _tmpfs_volume(opts: str) -> dict:
    return {"driver": "local", "driver_opts": {"type": "tmpfs", "device": "tmpfs", "o": opts}}


def _service(firm: str, scratch_volume: str, home_volume: str, bulletin_volume: str, inbox_volume: str) -> dict:
    svc: dict = {
        "build": {"context": ".", "dockerfile_inline": DOCKERFILE},
        "image": IMAGE,
    }
    if firm == "A":
        svc["x-default"] = True
    svc.update({
        "init": True,
        "network_mode": "none",
        "read_only": True,
        "hostname": HOSTNAME,
        "working_dir": HOME,
        "mem_limit": "256m",
        "cpus": 0.5,
        "pids_limit": 64,
        "security_opt": ["no-new-privileges:true"],
        "cap_drop": ["ALL"],
        "cap_add": list(CAP_ADD),
        "tmpfs": [f"/tmp:size={TMP_SIZE},nr_inodes={TMP_INODES},mode=1777",
                  f"/dev/shm:size={SHM_SIZE},nr_inodes={SHM_INODES},mode=1777"],
        "volumes": [
            {"type": "volume", "source": home_volume, "target": HOME, "volume": {"nocopy": True}},
            {"type": "volume", "source": scratch_volume, "target": SCRATCH, "volume": {"nocopy": True}},
            {"type": "volume", "source": bulletin_volume, "target": BULLETIN, "volume": {"nocopy": True}},
            {"type": "volume", "source": inbox_volume, "target": INBOX, "volume": {"nocopy": True}},
        ],
        "command": ["/bin/sh", "-c", INIT_SCRIPT + "; exec sleep infinity"],
    })
    return svc


def compose_dict(shared: bool) -> dict:
    """The compose document as a dict (exposed for tests and inspection)."""
    scratch = {"A": "scratch", "B": "scratch"} if shared else {"A": "scratch_a", "B": "scratch_b"}
    home = {"A": "home_a", "B": "home_b"}
    bulletin = {"A": "bulletin_a", "B": "bulletin_b"}
    inbox = {"A": "inbox_a", "B": "inbox_b"}
    volumes: dict = {}
    for firm in FIRMS:
        uid = UID[firm]
        volumes[home[firm]] = _tmpfs_volume(f"size={HOME_SIZE},nr_inodes={HOME_INODES},uid={uid},gid={uid},mode=0700,nosuid,nodev")
    for name in sorted(set(scratch.values())):
        volumes[name] = _tmpfs_volume(f"size={SCRATCH_SIZE},nr_inodes={SCRATCH_INODES},mode=1777,nosuid,nodev")
    for name in bulletin.values():
        volumes[name] = _tmpfs_volume(
            f"size={BULLETIN_SIZE},nr_inodes={BULLETIN_INODES},uid=0,gid=0,mode=0755,nosuid,nodev")
    for name in inbox.values():
        volumes[name] = _tmpfs_volume(
            f"size={INBOX_SIZE},nr_inodes={INBOX_INODES},uid=0,gid=0,mode=0755,nosuid,nodev")
    return {
        # Inspect cannot verify volumes that carry driver_opts (such a volume could be a disguised host bind), so it
        # refuses them unless this literal opt-in is set. Every volume here is a size-bounded tmpfs with no host
        # access; validate_compose() checks exactly that, fail-closed, before the file is written.
        "x-inspect_docker_sandbox": {"unsafe_allow_writable_host_mounts": True},
        "services": {SERVICE[f]: _service(f, scratch[f], home[f], bulletin[f], inbox[f]) for f in FIRMS},
        "volumes": volumes,
    }


_FORBIDDEN_SERVICE_KEYS = ("privileged", "devices", "configs", "secrets", "volumes_from", "volume_driver", "ports",
                           "network", "networks", "pid", "ipc", "userns_mode", "cgroup_parent", "extra_hosts",
                           "env_file", "environment", "container_name")
_ALLOWED_CAPS = set(CAP_ADD)


def validate_compose(doc: dict) -> None:
    """Fail-closed check standing in for Inspect's mount validation (which the opt-in above disables).

    Raises ValueError unless: no service has host mounts, devices, privileges or extra capabilities; every mount is a
    named volume declared here; and every volume is a local-driver tmpfs (device tmpfs, a size limit, no bind option).
    """
    def bad(msg: str) -> None:
        raise ValueError(f"unsafe compose document: {msg}")

    volumes = doc.get("volumes") or {}
    for name, vol in volumes.items():
        opts = (vol or {}).get("driver_opts") or {}
        o = str(opts.get("o", ""))
        if (vol or {}).get("external") or (vol or {}).get("name") or (vol or {}).get("driver", "local") != "local":
            bad(f"volume {name} is external, renamed or not local")
        if opts.get("type") != "tmpfs" or opts.get("device") != "tmpfs" or set(opts) - {"type", "device", "o"}:
            bad(f"volume {name} is not a plain tmpfs")
        if "size=" not in o or "nr_inodes=" not in o or "bind" in o or "/" in o:
            bad(f"volume {name} has unbounded or host-referencing options {o!r}")
    for sname, svc in (doc.get("services") or {}).items():
        for key in _FORBIDDEN_SERVICE_KEYS:
            if key in svc:
                bad(f"service {sname} sets {key}")
        if svc.get("network_mode") != "none" or svc.get("read_only") is not True:
            bad(f"service {sname} is networked or has a writable root")
        if svc.get("cap_drop") != ["ALL"] or not set(svc.get("cap_add", [])) <= _ALLOWED_CAPS:
            bad(f"service {sname} capabilities are not minimal")
        if "no-new-privileges:true" not in (svc.get("security_opt") or []):
            bad(f"service {sname} lacks no-new-privileges")
        for m in svc.get("volumes", []):
            if not isinstance(m, dict) or m.get("type") != "volume" or m.get("source") not in volumes:
                bad(f"service {sname} has a mount that is not a declared tmpfs volume: {m!r}")
        mounts = {m.get("target"): m.get("source") for m in svc.get("volumes", [])}
        if len(mounts) != 4 or set(mounts) != {HOME, SCRATCH, BULLETIN, INBOX}:
            bad(f"service {sname} must mount separate home, scratch, bulletin and inbox volumes")
        if len(set(mounts.values())) != len(mounts):
            bad(f"service {sname} aliases distinct mount paths to the same volume")
        if mounts[BULLETIN] == mounts[SCRATCH]:
            bad(f"service {sname} bulletin shares the agent-writable scratch quota")
        if mounts[INBOX] == mounts[HOME]:
            bad(f"service {sname} inbox shares the agent-writable home quota")
        for path, size, inodes in ((BULLETIN, BULLETIN_SIZE, BULLETIN_INODES),
                                  (INBOX, INBOX_SIZE, INBOX_INODES)):
            opts = str(volumes[mounts[path]]["driver_opts"].get("o", ""))
            if not all(x in opts for x in (f"size={size}", f"nr_inodes={inodes}",
                                            "uid=0", "gid=0", "mode=0755")):
                bad(f"service {sname} {path} volume is not bounded and root-owned")
        for t in svc.get("tmpfs", []):
            if "size=" not in str(t) or "nr_inodes=" not in str(t):
                bad(f"service {sname} has an unbounded tmpfs {t!r}")
        tm = {str(t).split(":", 1)[0]: str(t) for t in svc.get("tmpfs", [])}
        if len(tm) != len(svc.get("tmpfs", [])) or set(tm) != {"/tmp", "/dev/shm"}:
            bad(f"service {sname} must bound both /tmp and /dev/shm")
    services = doc.get("services") or {}
    if set(services) != {SERVICE[f] for f in FIRMS}:
        bad("exactly the two intended firm services are required")
    private_sources = []
    for target in (HOME, BULLETIN, INBOX):
        a = next(m["source"] for m in services[SERVICE["A"]]["volumes"] if m["target"] == target)
        b = next(m["source"] for m in services[SERVICE["B"]]["volumes"] if m["target"] == target)
        private_sources.extend((a, b))
        if a == b:
            bad(f"the firms must not share {target}")
    scratch_sources = {next(m["source"] for m in services[SERVICE[f]]["volumes"] if m["target"] == SCRATCH)
                       for f in FIRMS}
    if len(set(private_sources)) != len(private_sources) or set(private_sources) & scratch_sources:
        bad("private home, inbox or bulletin volumes alias another mount")


def write_compose(dirpath: str, shared: bool) -> str:
    """Write the compose file (shared or private scratch) into `dirpath` and return its path.

    The directory is created if needed. It is also the (empty) build context of the derived image, and nothing in it
    is mounted into the containers.
    """
    doc = compose_dict(shared)
    validate_compose(doc)
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    path = d / COMPOSE_NAME
    header = ("# Generated by harness/sandbox_env.py. "
              f"{'SHR: one /srv/scratch volume shared by both firms' if shared else 'ISO: a private /srv/scratch per firm'}"
              ".\n")
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(header + yaml.safe_dump(doc, sort_keys=False))
    os.replace(tmp, path)
    return str(path)
