"""A docker sandbox records where its bind mounts land in the container.

Strict delivery with ``gateway.fetch_own_engine_dir_files`` sends a file the session wrote in the sandbox's engine
folder but never one the engine mounted there. The container's own record of its mounts, taken when its run
arguments were built, is what tells the two apart: names and the per-turn credential registry cannot.
"""

from tests.tools.test_docker_environment import _make_dummy_env, _mock_subprocess_run
from tools.environments import docker as docker_env

ENGINE_BINDS = ["-v", "/srv/hermes/my_token.json:/root/.hermes/my_token.json:ro",
                "-v", "/srv/hermes/skills:/root/.hermes/skills:ro"]


def _bound_container_paths(run_args: list[str]) -> set[str]:
    return {spec.split(":")[1] for flag, spec in zip(run_args, run_args[1:]) if flag == "-v"}


def test_a_docker_sandbox_records_the_container_path_of_every_bind_it_starts_with(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_readonly_skill_mount_args", lambda: list(ENGINE_BINDS))
    calls = _mock_subprocess_run(monkeypatch)

    env = _make_dummy_env(volumes=["/srv/data:/data:ro"])

    run_args = next(cmd for cmd, _ in calls if isinstance(cmd, list) and len(cmd) >= 2 and cmd[1] == "run")
    assert set(env._engine_mount_targets) == _bound_container_paths(run_args)
    assert {"/root/.hermes/my_token.json", "/root/.hermes/skills", "/data"} <= set(env._engine_mount_targets)


def test_bind_targets_are_read_from_each_volume_spec():
    from tools.environments.docker_mount_targets import bind_mount_targets

    assert bind_mount_targets(["-v", "/a:/b", "--tmpfs", "/tmp:rw", "-v", "/c:/root/.hermes/d/../e:ro",
                               "-v", r"C:\Users\me:/workspace:rw", "-v", "named-volume"]) == (
        "/b", "/root/.hermes/e", "/workspace")
