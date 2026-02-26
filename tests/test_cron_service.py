from pathlib import Path

from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule


def test_add_job_preserves_unicode_message(tmp_path) -> None:
    store_path = Path(tmp_path) / "cron.json"
    service = CronService(store_path)

    service.add_job(
        name="unicode",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="提醒：天气☀️很好，注意补水",
    )

    raw = store_path.read_text(encoding="utf-8")
    assert "提醒：天气☀️很好，注意补水" in raw
    assert "\\u" not in raw


def test_load_store_normalizes_legacy_escaped_unicode(tmp_path) -> None:
    store_path = Path(tmp_path) / "cron.json"
    store_path.write_text(
        '{"version":1,"jobs":[{"id":"job1","name":"legacy","enabled":true,"schedule":{"kind":"every","everyMs":60000},"payload":{"kind":"agent_turn","message":"\\u63d0\\u9192\\uff1a\\u5929\\u6c14\\u5f88\\u597d"},"state":{},"createdAtMs":1,"updatedAtMs":1,"deleteAfterRun":false}]}\n',
        encoding="utf-8",
    )

    service = CronService(store_path)
    jobs = service.list_jobs(include_disabled=True)

    assert jobs[0].payload.message == "提醒：天气很好"

    raw = store_path.read_text(encoding="utf-8")
    assert "提醒：天气很好" in raw
    assert "\\u63d0" not in raw
