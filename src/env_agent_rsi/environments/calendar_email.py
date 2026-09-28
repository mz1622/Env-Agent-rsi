from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Mapping

from env_agent_rsi.core.protocol import EnvResponse, JsonObject
from env_agent_rsi.core.verifier import StateVerifier
from env_agent_rsi.environments.base import StatefulTaskEnv, tool


CALENDAR_EMAIL_TOOLS = [
    tool(
        "search_contacts",
        "Search workplace contacts by name.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    tool(
        "get_calendar",
        "Read busy events for the owner or one contact on a date.",
        {
            "email": {"type": "string"},
            "date": {"type": "string", "description": "YYYY-MM-DD"},
        },
        ["email", "date"],
    ),
    tool(
        "find_free_slots",
        "Find common free slots in working hours for all attendees.",
        {
            "emails": {"type": "array", "items": {"type": "string"}},
            "date": {"type": "string"},
            "duration_minutes": {"type": "integer"},
        },
        ["emails", "date", "duration_minutes"],
    ),
    tool(
        "create_event",
        "Create one calendar event.",
        {
            "title": {"type": "string"},
            "attendee_emails": {"type": "array", "items": {"type": "string"}},
            "start": {"type": "string", "description": "YYYY-MM-DDTHH:MM:SS"},
            "duration_minutes": {"type": "integer"},
        },
        ["title", "attendee_emails", "start", "duration_minutes"],
    ),
    tool(
        "get_event",
        "Read an event by id.",
        {"event_id": {"type": "string"}},
        ["event_id"],
    ),
    tool(
        "send_email",
        "Send one email immediately.",
        {
            "recipient": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        ["recipient", "subject", "body"],
    ),
    tool("finish", "Finish the task and run the independent verifier.", {}),
]


class CalendarEmailEnv(StatefulTaskEnv):
    def __init__(
        self,
        *,
        verifier: StateVerifier,
        task_id: str = "workbench_multidomain_adapted_151",
        instruction: str = (
            "Leila has overdue tasks. Find Leila Azizi's contact, book a 30-minute "
            "meeting titled 'Catch up on overdue tasks' at the earliest time both of "
            "you are free on 2023-12-01, and send Leila exactly one email with subject "
            "'Discuss overdue tasks' and body \"I noticed you have a few overdue tasks "
            "- let's catch up tomorrow.\" Confirm the event, then finish."
        ),
    ) -> None:
        super().__init__(
            verifier=verifier,
            task_id=task_id,
            instruction=instruction,
            tool_schemas=CALENDAR_EMAIL_TOOLS,
        )

    def initial_state(self, seed: int, options: Mapping[str, Any]) -> JsonObject:
        del seed, options
        return {
            "owner_email": "me@atlas.com",
            "working_hours": {"start": "09:00", "end": "17:00"},
            "contacts": [
                {"id": "P-10", "name": "Leila Azizi", "email": "leila.azizi@atlas.com"},
                {
                    "id": "P-11",
                    "name": "Leila Martin",
                    "email": "leila.martin@atlas.com",
                },
            ],
            "busy": {
                "me@atlas.com": [
                    {"start": "2023-12-01T09:00:00", "end": "2023-12-01T13:00:00"}
                ],
                "leila.azizi@atlas.com": [
                    {"start": "2023-12-01T13:30:00", "end": "2023-12-01T14:30:00"}
                ],
                "leila.martin@atlas.com": [],
            },
            "events": {},
            "emails": [],
        }

    def handlers(self):
        return {
            "search_contacts": self._search_contacts,
            "get_calendar": self._get_calendar,
            "find_free_slots": self._find_free_slots,
            "create_event": self._create_event,
            "get_event": self._get_event,
            "send_email": self._send_email,
            "finish": self.finish,
        }

    def _search_contacts(self, arguments: JsonObject) -> EnvResponse:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return self.error("INVALID_ARGUMENT", "query must be a non-empty string")
        needle = query.casefold()
        contacts = [
            deepcopy(contact)
            for contact in self.state["contacts"]
            if needle in contact["name"].casefold()
        ]
        return self.success({"contacts": contacts}, event="search_contacts")

    def _get_calendar(self, arguments: JsonObject) -> EnvResponse:
        email = arguments.get("email")
        date = arguments.get("date")
        if email not in self.state["busy"]:
            return self.error("NOT_FOUND", "calendar was not found")
        busy = [
            deepcopy(interval)
            for interval in self.state["busy"][email]
            if interval["start"].startswith(str(date))
        ]
        return self.success(
            {"email": email, "date": date, "busy": busy}, event="get_calendar"
        )

    @staticmethod
    def _parse(value: str) -> datetime:
        return datetime.fromisoformat(value)

    def _all_busy(self, email: str) -> list[tuple[datetime, datetime]]:
        intervals = self.state["busy"].get(email, [])
        events = [
            {"start": event["start"], "end": event["end"]}
            for event in self.state["events"].values()
            if email == self.state["owner_email"] or email in event["attendee_emails"]
        ]
        return [
            (self._parse(x["start"]), self._parse(x["end"])) for x in intervals + events
        ]

    def _find_free_slots(self, arguments: JsonObject) -> EnvResponse:
        emails = arguments.get("emails")
        date = arguments.get("date")
        duration = arguments.get("duration_minutes")
        if (
            not isinstance(emails, list)
            or not emails
            or not all(x in self.state["busy"] for x in emails)
        ):
            return self.error(
                "INVALID_ARGUMENT", "emails must identify known calendars"
            )
        if not isinstance(duration, int) or duration <= 0:
            return self.error("INVALID_ARGUMENT", "duration_minutes must be positive")
        try:
            cursor = self._parse(f"{date}T{self.state['working_hours']['start']}:00")
            end = self._parse(f"{date}T{self.state['working_hours']['end']}:00")
        except (TypeError, ValueError):
            return self.error("INVALID_ARGUMENT", "date must use YYYY-MM-DD")
        delta = timedelta(minutes=duration)
        slots = []
        while cursor + delta <= end:
            slot_end = cursor + delta
            conflict = any(
                cursor < busy_end and slot_end > busy_start
                for email in emails
                for busy_start, busy_end in self._all_busy(email)
            )
            if not conflict:
                slots.append({"start": cursor.isoformat(), "end": slot_end.isoformat()})
            cursor += timedelta(minutes=30)
        return self.success({"slots": slots}, event="find_free_slots")

    def _create_event(self, arguments: JsonObject) -> EnvResponse:
        title = arguments.get("title")
        attendees = arguments.get("attendee_emails")
        start_value = arguments.get("start")
        duration = arguments.get("duration_minutes")
        if not isinstance(title, str) or not title:
            return self.error("INVALID_ARGUMENT", "title must be non-empty")
        if not isinstance(attendees, list) or not attendees:
            return self.error("INVALID_ARGUMENT", "attendee_emails must be non-empty")
        if not isinstance(duration, int) or duration <= 0:
            return self.error("INVALID_ARGUMENT", "duration_minutes must be positive")
        try:
            start = self._parse(str(start_value))
        except ValueError:
            return self.error("INVALID_ARGUMENT", "start must be an ISO datetime")
        end = start + timedelta(minutes=duration)
        participants = [self.state["owner_email"], *attendees]
        if any(
            start < busy_end and end > busy_start
            for email in participants
            for busy_start, busy_end in self._all_busy(email)
        ):
            return self.error("CALENDAR_CONFLICT", "one or more attendees are busy")
        event_id = f"E-{len(self.state['events']) + 1}"
        event = {
            "id": event_id,
            "title": title,
            "attendee_emails": list(attendees),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "duration_minutes": duration,
        }
        self.state["events"][event_id] = event
        return self.success(
            {"event": deepcopy(event)}, event="create_event", state_changed=True
        )

    def _get_event(self, arguments: JsonObject) -> EnvResponse:
        event = self.state["events"].get(arguments.get("event_id"))
        if not event:
            return self.error("NOT_FOUND", "event was not found")
        return self.success({"event": deepcopy(event)}, event="get_event")

    def _send_email(self, arguments: JsonObject) -> EnvResponse:
        recipient = arguments.get("recipient")
        subject = arguments.get("subject")
        body = arguments.get("body")
        if not all(
            isinstance(value, str) and value for value in (recipient, subject, body)
        ):
            return self.error(
                "INVALID_ARGUMENT", "recipient, subject, and body are required"
            )
        email = {
            "id": f"M-{len(self.state['emails']) + 1}",
            "recipient": recipient,
            "subject": subject,
            "body": body,
        }
        self.state["emails"].append(email)
        return self.success(
            {"email": deepcopy(email)}, event="send_email", state_changed=True
        )
