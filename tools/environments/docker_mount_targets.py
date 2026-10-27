"""Where a docker sandbox's bind mounts land in the container (``gateway.fetch_own_engine_dir_files``)."""

import posixpath


def bind_mount_targets(volume_args: list[str]) -> tuple[str, ...]:
    """Container paths of the ``-v host:container[:options]`` binds in *volume_args*; a named volume has none."""
    targets: list[str] = []
    for flag, spec in zip(volume_args, volume_args[1:]):
        if flag not in ("-v", "--volume"):
            continue
        container = next((part for part in spec.split(":")[1:] if part.startswith("/")), None)
        if container:
            targets.append(posixpath.normpath(container))
    return tuple(targets)
