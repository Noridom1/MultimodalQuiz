from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from src.application.services.run_context import RunContext
from src.utils.io import append_jsonl

logger = logging.getLogger(__name__)


class RunLogService:
    def log_event(
        self,
        context: RunContext,
        stage: str,
        event: str,
        message: str,
        **details: Any,
    ) -> None:
        append_jsonl(
            context.log_path,
            {
                "timestamp": dt.datetime.utcnow().isoformat() + "Z",
                "run_id": context.run_id,
                "stage": stage,
                "event": event,
                "message": message,
                "details": details,
            },
        )
        log_message = (
            f"run_id={context.run_id} stage={stage} event={event} message={message} details={details}"
        )
        if event == "failed":
            logger.error(log_message)
        else:
            logger.info(log_message)
