from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from env_agent_rsi.core.protocol import EnvResponse, JsonObject
from env_agent_rsi.core.verifier import StateVerifier
from env_agent_rsi.environments.base import StatefulTaskEnv, tool


ORDER_TOOLS = [
    tool(
        "authenticate_user",
        "Authenticate one customer by email or by first_name, last_name, and zip.",
        {
            "email": {"type": "string"},
            "first_name": {"type": "string"},
            "last_name": {"type": "string"},
            "zip": {"type": "string"},
        },
    ),
    tool(
        "search_orders",
        "List orders belonging to the authenticated customer.",
        {"status": {"type": "string"}},
    ),
    tool(
        "get_order",
        "Read one order, including allowed replacements for each item.",
        {"order_id": {"type": "string"}},
        ["order_id"],
    ),
    tool(
        "get_payment_method",
        "Read a payment method attached to the authenticated customer.",
        {"payment_method_id": {"type": "string"}},
        ["payment_method_id"],
    ),
    tool(
        "modify_order_items",
        "Replace items in one pending order. Each replacement must be allowed.",
        {
            "order_id": {"type": "string"},
            "replacements": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        },
        ["order_id", "replacements"],
    ),
    tool(
        "cancel_order",
        "Cancel a pending order and create a refund to its original payment method.",
        {
            "order_id": {"type": "string"},
            "reason": {
                "type": "string",
                "enum": ["no longer needed", "ordered by mistake"],
            },
        },
        ["order_id", "reason"],
    ),
    tool(
        "get_refund",
        "Read a refund by order id.",
        {"order_id": {"type": "string"}},
        ["order_id"],
    ),
    tool("finish", "Finish the task and run the independent verifier.", {}),
]


class OrderLifecycleEnv(StatefulTaskEnv):
    def __init__(
        self,
        *,
        verifier: StateVerifier,
        task_id: str = "tau_retail_adapted_66",
        instruction: str = (
            "You are Aarav Lee in Phoenix, ZIP 85025. Find your pending order "
            "containing a luggage set. You want to replace it with a coat. If the "
            "order says that replacement is not allowed, cancel the entire order "
            "with reason 'no longer needed'. Confirm the refund, then finish. Do not "
            "change any other order."
        ),
    ) -> None:
        super().__init__(
            verifier=verifier,
            task_id=task_id,
            instruction=instruction,
            tool_schemas=ORDER_TOOLS,
        )

    def initial_state(self, seed: int, options: Mapping[str, Any]) -> JsonObject:
        del seed, options
        return {
            "users": [
                {
                    "id": "aarav_lee_1982",
                    "first_name": "Aarav",
                    "last_name": "Lee",
                    "zip": "85025",
                    "email": "aarav.lee@example.com",
                },
                {
                    "id": "aarav_lee_7711",
                    "first_name": "Aarav",
                    "last_name": "Lee",
                    "zip": "85026",
                    "email": "other.aarav@example.com",
                },
            ],
            "authenticated_user_id": None,
            "orders": {
                "O-1007": {
                    "id": "O-1007",
                    "user_id": "aarav_lee_1982",
                    "status": "pending",
                    "items": [
                        {
                            "item_id": "LI-44",
                            "product": "luggage set",
                            "price": 189.0,
                            "allowed_replacements": [],
                        }
                    ],
                    "total": 189.0,
                    "payment_method_id": "gift_card_19",
                },
                "O-1008": {
                    "id": "O-1008",
                    "user_id": "aarav_lee_1982",
                    "status": "delivered",
                    "items": [
                        {
                            "item_id": "CO-10",
                            "product": "winter coat",
                            "price": 129.0,
                            "allowed_replacements": [],
                        }
                    ],
                    "total": 129.0,
                    "payment_method_id": "card_32",
                },
            },
            "payment_methods": {
                "gift_card_19": {
                    "id": "gift_card_19",
                    "user_id": "aarav_lee_1982",
                    "type": "gift_card",
                    "last_four": "0019",
                },
                "card_32": {
                    "id": "card_32",
                    "user_id": "aarav_lee_1982",
                    "type": "credit_card",
                    "last_four": "3232",
                },
            },
            "refunds": {},
        }

    def handlers(self):
        return {
            "authenticate_user": self._authenticate_user,
            "search_orders": self._search_orders,
            "get_order": self._get_order,
            "get_payment_method": self._get_payment_method,
            "modify_order_items": self._modify_order_items,
            "cancel_order": self._cancel_order,
            "get_refund": self._get_refund,
            "finish": self.finish,
        }

    def _authenticate_user(self, arguments: JsonObject) -> EnvResponse:
        for user in self.state["users"]:
            email_match = arguments.get("email") == user["email"]
            identity_match = all(
                arguments.get(field) == user[field]
                for field in ("first_name", "last_name", "zip")
            )
            if email_match or identity_match:
                self.state["authenticated_user_id"] = user["id"]
                return self.success(
                    {"user": deepcopy(user)},
                    event="authenticate_user",
                    state_changed=True,
                )
        return self.error("AUTHENTICATION_FAILED", "customer identity did not match")

    def _require_auth(self) -> str | None:
        return self.state["authenticated_user_id"]

    def _search_orders(self, arguments: JsonObject) -> EnvResponse:
        user_id = self._require_auth()
        if not user_id:
            return self.error(
                "AUTHENTICATION_REQUIRED", "authenticate the customer first"
            )
        status = arguments.get("status")
        orders = [
            deepcopy(order)
            for order in self.state["orders"].values()
            if order["user_id"] == user_id
            and (status is None or order["status"] == status)
        ]
        return self.success({"orders": orders}, event="search_orders")

    def _owned_order(self, order_id: Any) -> JsonObject | None:
        order = self.state["orders"].get(str(order_id))
        if order and order["user_id"] == self._require_auth():
            return order
        return None

    def _get_order(self, arguments: JsonObject) -> EnvResponse:
        if not self._require_auth():
            return self.error(
                "AUTHENTICATION_REQUIRED", "authenticate the customer first"
            )
        order = self._owned_order(arguments.get("order_id"))
        if not order:
            return self.error("NOT_FOUND", "order was not found for this customer")
        return self.success({"order": deepcopy(order)}, event="get_order")

    def _get_payment_method(self, arguments: JsonObject) -> EnvResponse:
        if not self._require_auth():
            return self.error(
                "AUTHENTICATION_REQUIRED", "authenticate the customer first"
            )
        method = self.state["payment_methods"].get(arguments.get("payment_method_id"))
        if not method or method["user_id"] != self._require_auth():
            return self.error("NOT_FOUND", "payment method was not found")
        return self.success(
            {"payment_method": deepcopy(method)}, event="get_payment_method"
        )

    def _modify_order_items(self, arguments: JsonObject) -> EnvResponse:
        if not self._require_auth():
            return self.error(
                "AUTHENTICATION_REQUIRED", "authenticate the customer first"
            )
        order = self._owned_order(arguments.get("order_id"))
        if not order:
            return self.error("NOT_FOUND", "order was not found for this customer")
        if order["status"] != "pending":
            return self.error(
                "ORDER_NOT_PENDING", "only pending orders can be modified"
            )
        replacements = arguments.get("replacements")
        if not isinstance(replacements, dict) or not replacements:
            return self.error(
                "INVALID_ARGUMENT", "replacements must be a non-empty object"
            )
        by_id = {item["item_id"]: item for item in order["items"]}
        for old_id, new_id in replacements.items():
            if (
                old_id not in by_id
                or new_id not in by_id[old_id]["allowed_replacements"]
            ):
                return self.error(
                    "REPLACEMENT_NOT_ALLOWED",
                    f"item {old_id!r} cannot be replaced with {new_id!r}",
                )
        for old_id, new_id in replacements.items():
            by_id[old_id]["item_id"] = new_id
        return self.success(
            {"order": deepcopy(order)}, event="modify_order_items", state_changed=True
        )

    def _cancel_order(self, arguments: JsonObject) -> EnvResponse:
        if not self._require_auth():
            return self.error(
                "AUTHENTICATION_REQUIRED", "authenticate the customer first"
            )
        order = self._owned_order(arguments.get("order_id"))
        if not order:
            return self.error("NOT_FOUND", "order was not found for this customer")
        if order["status"] != "pending":
            return self.error(
                "ORDER_NOT_PENDING", "only pending orders can be cancelled"
            )
        reason = arguments.get("reason")
        if reason not in {"no longer needed", "ordered by mistake"}:
            return self.error("INVALID_REASON", "use an allowed cancellation reason")
        order["status"] = "cancelled"
        order["cancellation_reason"] = reason
        refund = {
            "id": "R-O-1007",
            "order_id": order["id"],
            "amount": order["total"],
            "payment_method_id": order["payment_method_id"],
            "status": "completed",
        }
        self.state["refunds"][order["id"]] = refund
        return self.success(
            {"order": deepcopy(order), "refund": deepcopy(refund)},
            event="cancel_order",
            state_changed=True,
        )

    def _get_refund(self, arguments: JsonObject) -> EnvResponse:
        if not self._require_auth():
            return self.error(
                "AUTHENTICATION_REQUIRED", "authenticate the customer first"
            )
        refund = self.state["refunds"].get(arguments.get("order_id"))
        if not refund:
            return self.error("NOT_FOUND", "refund was not found")
        return self.success({"refund": deepcopy(refund)}, event="get_refund")
