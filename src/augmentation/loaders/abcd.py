"""ABCD dataset loader with user goal from flow/subflow."""

import json
from pathlib import Path
from typing import Iterator

from augmentation.schema import Dialogue, Goal, StructuredGoal
from .base import BaseLoader


class ABCDLoader(BaseLoader):
    """Load ABCD dialogues with goals from scenario flow/subflow."""

    def __init__(self, data_dir: Path, split: str = "train"):
        super().__init__(data_dir, split)
        self._data_cache: dict | None = None

    @property
    def name(self) -> str:
        return "abcd"

    def _load_data(self) -> dict:
        """Load and cache ABCD data."""
        if self._data_cache is None:
            data_path = self.data_dir / "data" / "abcd_v1.1.json"
            with open(data_path, "r") as f:
                self._data_cache = json.load(f)
        return self._data_cache

    def load(self) -> Iterator[Dialogue]:
        """Yield ABCD dialogues."""
        data = self._load_data()
        dialogues = data.get(self.split, [])

        for dlg in dialogues:
            dlg_id = dlg.get("convo_id", "")
            scenario = dlg.get("scenario", {})

            # Extract goal from scenario
            goal = self._extract_goal(scenario)

            # Extract turns from delexed conversation
            turns = []
            for turn_data in dlg.get("delexed", []):
                speaker = turn_data.get("speaker", "").lower()
                role = "user" if speaker == "customer" else "assistant"
                text = turn_data.get("text", "")
                turns.append({"role": role, "text": text})

            # Build state from scenario personal/order info
            state = self._extract_state(scenario)

            yield Dialogue(
                id=dlg_id,
                source=self.name,
                turns=turns,
                goal=goal,
                state=state,
                metadata={"scenario": scenario},
            )

    def _extract_goal(self, scenario: dict) -> Goal:
        """Extract user goal from scenario."""
        flow = scenario.get("flow", "")
        subflow = scenario.get("subflow", "")
        personal = scenario.get("personal", {})
        order = scenario.get("order", {})
        product = scenario.get("product", {})

        # Build text goal
        text = self._build_goal_text(flow, subflow, personal, order, product)

        # Build structured goal
        structured = self._build_structured_goal(flow, subflow, personal, order, product)

        return Goal(text=text, structured=structured)

    def _build_goal_text(
        self,
        flow: str,
        subflow: str,
        personal: dict,
        order: dict,
        product: dict,
    ) -> str:
        """Build natural language goal."""
        parts = []

        # Convert flow/subflow to natural description
        flow_phrases = {
            "order_issue": "an issue with your order",
            "product_defect": "a defective product",
            "shipping": "a shipping concern",
            "storewide_query": "a question about the store",
            "account_access": "accessing your account",
            "single_item_query": "a question about a specific item",
            "subscription_inquiry": "a subscription question",
            "purchase_dispute": "a purchase dispute",
        }
        
        subflow_phrases = {
            "missing_item": "Some items are missing from your order.",
            "late_delivery": "Your delivery is late.",
            "wrong_item": "You received the wrong item.",
            "damaged": "Your item arrived damaged.",
            "refund": "You want to request a refund.",
            "exchange": "You want to exchange an item.",
            "cancel_order": "You want to cancel your order.",
            "track_order": "You want to track your order.",
            "change_address": "You need to change your shipping address.",
            "promo_code": "You have a question about a promo code.",
        }
        
        flow_text = flow_phrases.get(flow, flow.replace("_", " "))
        subflow_text = subflow_phrases.get(subflow, subflow.replace("_", " ") + ".")
        
        parts.append(f"You are calling customer service about {flow_text}. {subflow_text}")

        # Customer context
        if personal.get("customer_name"):
            parts.append(f"Your name is {personal['customer_name']}.")

        if order.get("order_id"):
            parts.append(f"Your order ID is {order['order_id']}.")

        if order.get("product_names"):
            products = ", ".join(order["product_names"][:2])
            parts.append(f"The order includes {products}.")

        return " ".join(parts)

    def _build_structured_goal(
        self,
        flow: str,
        subflow: str,
        personal: dict,
        order: dict,
        product: dict,
    ) -> StructuredGoal:
        """Build structured goal."""
        # Combine all slots
        slots = {}
        slots.update(personal)
        slots.update({f"order_{k}": v for k, v in order.items()})
        slots.update({f"product_{k}": v for k, v in product.items()})

        # Filter out complex nested values
        slots = {k: v for k, v in slots.items() if isinstance(v, (str, int, float))}

        return StructuredGoal(
            domains=["customer_service"],
            intents=[{
                "domain": "customer_service",
                "intent": f"{flow}.{subflow}",
                "slots": slots,
                "requests": [],
            }],
        )

    def _extract_state(self, scenario: dict) -> dict:
        """Extract dialogue state from scenario."""
        state = {}
        personal = scenario.get("personal", {})
        order = scenario.get("order", {})

        state["customer_service"] = {}
        state["customer_service"].update(personal)
        state["customer_service"].update(order)

        return state
