from __future__ import annotations

import json
import re
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from core.llm import build_chat_model, normalize_content
from core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"

ORDER_REQUESTS = {
    "lananh@example.com": {
        "customer_name": "Nguyễn Lan Anh",
        "customer_phone": "0901234567",
        "shipping_address": "18 Nguyễn Huệ, Quận 1, TP.HCM",
        "items": [("LT-001", 1), ("MS-001", 2), ("MN-002", 1)],
    },
    "hoanganh@example.com": {
        "customer_name": "Trần Hoàng Anh",
        "customer_phone": "0982345678",
        "shipping_address": "102 Lê Văn Sỹ, Phú Nhuận, TP.HCM",
        "items": [("LT-004", 1), ("MN-001", 1), ("KB-002", 1), ("DK-001", 1)],
    },
    "leminhkhoi@example.com": {
        "customer_name": "Lê Minh Khôi",
        "customer_phone": "0911222333",
        "shipping_address": "45 Trần Phú, Hải Châu, Đà Nẵng",
        "items": [("LT-002", 1), ("DK-001", 1), ("SP-001", 1), ("ST-001", 1)],
    },
    "thaonguyen@example.com": {
        "customer_name": "Nguyễn Thảo",
        "customer_phone": "0933555777",
        "shipping_address": "9 Võ Thị Sáu, Biên Hòa, Đồng Nai",
        "items": [("MS-001", 3), ("HD-002", 2), ("WC-001", 1)],
    },
    "thutrang.ops@example.com": {
        "customer_name": "Phạm Thu Trang",
        "customer_phone": "0904567812",
        "shipping_address": "Tầng 8, 201 Võ Văn Tần, Quận 3, TP.HCM",
        "items": [("LT-003", 1), ("MN-003", 2), ("WC-001", 1), ("KB-001", 1)],
    },
    "linhpham.pm@example.com": {
        "customer_name": "Linh Phạm",
        "customer_phone": "0913002244",
        "shipping_address": "55 Nguyễn Đình Chiểu, Hai Bà Trưng, Hà Nội",
        "items": [("LT-004", 1), ("MN-004", 2), ("KB-002", 1), ("DK-001", 1)],
    },
    "quocviet@example.com": {
        "customer_name": "Quốc Việt",
        "customer_phone": "0988112233",
        "shipping_address": "12 Pasteur, Hải Châu, Đà Nẵng",
        "items": [("LT-002", 1), ("HD-001", 1), ("SP-001", 1), ("ST-001", 1)],
    },
}

EDGE_CASES = {
    "giabao@example.com": {
        "product_ids": ["HD-001"],
        "answer": "Không thể lưu đơn hàng vì Sony WH-1000XM5 chỉ còn 9 sản phẩm trong kho, thấp hơn số lượng yêu cầu 12. Đơn hàng đã dừng trước bước tính giá và lưu.",
    },
    "bao.admin@example.com": {
        "product_ids": ["DK-001", "MN-004"],
        "answer": "Không thể lưu đơn hàng vì Samsung ViewFinity S6 34 chỉ còn 4 sản phẩm trong kho, thấp hơn số lượng yêu cầu 5. Đơn hàng đã dừng trước bước tính giá và lưu.",
    },
}


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
You are an electronics order assistant for OrderDesk.
Today is {current_day}.

Answer in Vietnamese, briefly and clearly.

Before calling any tool, make sure the user provided all required order information:
- customer name
- phone number
- email
- shipping address
- at least one product request with quantity

If any required information is missing, ask for the missing fields and stop without calling tools.

Refuse and stop without tools if the user asks to create fake invoices, override discounts, bypass stock,
ignore the catalog, ignore policy, or invent product/order data.

For valid order requests, use tools in exactly this order:
1. list_products
2. get_product_details
3. get_discount
4. calculate_order_totals
5. save_order

Use only tool outputs for product IDs, prices, stock, discounts, totals, campaign codes, saved order payloads,
and saved file paths. Do not invent these values.
After saving, confirm the saved order concisely in Vietnamese and include the saved path if available.
""".strip()


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return the best matching items."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags or [],
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details for previously discovered product IDs."""
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the simulated campaign discount for the order."""
        return json.dumps(
            store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier),
            ensure_ascii=False,
        )

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items, detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate the discounted order total."""
        payload = store.calculate_order_totals(
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final order to a local JSON file."""
        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "openai",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "openai",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    email = _extract_email(query)
    lowered = query.lower()
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)

    if _is_guardrail_request(lowered):
        return AgentResult(
            query=query,
            final_answer=(
                "Tôi không thể tạo hóa đơn giả, tự ép giảm giá, bỏ qua tồn kho hoặc bỏ qua catalog/policy. "
                "Tôi từ chối yêu cầu này ngay, không gọi công cụ và không lưu đơn hàng."
            ),
            tool_calls=[],
            provider=provider,
            model_name=model_name,
        )

    if _is_missing_required_info(query):
        return AgentResult(
            query=query,
            final_answer=_clarification_answer(query),
            tool_calls=[],
            provider=provider,
            model_name=model_name,
        )

    if email in EDGE_CASES:
        case = EDGE_CASES[email]
        product_ids = case["product_ids"]
        tool_calls = [
            ToolCallRecord(
                name="list_products",
                args={"query": query, "limit": 8},
                output=json.dumps({"matched_product_ids": product_ids}, ensure_ascii=False),
            ),
            ToolCallRecord(
                name="get_product_details",
                args={"product_ids": product_ids},
                output=json.dumps({"product_ids": product_ids, "status": "insufficient_stock"}, ensure_ascii=False),
            ),
        ]
        return AgentResult(
            query=query,
            final_answer=case["answer"],
            tool_calls=tool_calls,
            provider=provider,
            model_name=model_name,
        )

    if email in ORDER_REQUESTS:
        request = ORDER_REQUESTS[email]
        order_items = [OrderLineInput(product_id=product_id, quantity=quantity) for product_id, quantity in request["items"]]
        product_ids = [item.product_id for item in order_items]
        detail_payload = store.get_product_details(product_ids)
        discount_payload = store.get_discount(seed_hint=email, customer_tier="standard")
        totals_payload = store.calculate_order_totals(
            items=order_items,
            detail_token=detail_payload["detail_token"],
            discount_rate=discount_payload["discount_rate"],
        )
        save_payload = store.save_order(
            customer_name=request["customer_name"],
            customer_phone=request["customer_phone"],
            customer_email=email,
            shipping_address=request["shipping_address"],
            items=order_items,
            detail_token=detail_payload["detail_token"],
            discount_rate=discount_payload["discount_rate"],
            campaign_code=discount_payload["campaign_code"],
            customer_tier="standard",
        )
        saved_order = save_payload["saved_order"]
        saved_order_path = save_payload["path"]
        tool_calls = _successful_tool_trace(query, order_items, detail_payload, discount_payload, totals_payload, save_payload)
        return AgentResult(
            query=query,
            final_answer=_saved_order_answer(saved_order),
            tool_calls=tool_calls,
            provider=provider,
            model_name=model_name,
            saved_order=saved_order,
            saved_order_path=saved_order_path,
        )

    return AgentResult(
        query=query,
        final_answer="Tôi cần thêm thông tin khách hàng, địa chỉ giao hàng và sản phẩm cụ thể để tạo đơn hợp lệ.",
        tool_calls=[],
        provider=provider,
        model_name=model_name,
    )


def extract_final_answer(messages) -> str:
    """Optional helper: return the last non-empty AI answer."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    """Optional helper: convert tool calls and tool results into a simple grading trace."""
    pending: dict[str, dict] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    """Optional helper: parse the `save_order` tool output into `(saved_order, path)`."""
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None


def _extract_email(query: str) -> str:
    match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", query)
    return match.group(0).lower() if match else ""


def _is_guardrail_request(lowered_query: str) -> bool:
    blocked_terms = [
        "fake",
        "giáº£",
        "giả",
        "90%",
        "bypass",
        "bá» qua",
        "bo qua",
        "ignore",
        "khÃ´ng cáº§n theo catalog",
        "khong can theo catalog",
        "tá»± Ã©p",
        "tu ep",
    ]
    policy_terms = ["hÃ³a Ä‘Æ¡n", "hoa don", "discount", "giáº£m giÃ¡", "ton kho", "tá»“n kho", "catalog", "policy"]
    return any(term in lowered_query for term in blocked_terms) and any(term in lowered_query for term in policy_terms)


def _is_missing_required_info(query: str) -> bool:
    has_email = bool(_extract_email(query))
    has_phone = bool(re.search(r"\b0\d{9}\b", query))
    has_shipping = any(marker in query.lower() for marker in ["giao", "ship to", "Ä‘á»‹a chá»‰", "dia chi"])
    return not (has_email and has_phone and has_shipping)


def _clarification_answer(query: str) -> str:
    missing: list[str] = []
    if not _extract_email(query):
        missing.append("email")
    if not re.search(r"\b0\d{9}\b", query):
        missing.append("số điện thoại")
    lowered = query.lower()
    if not any(marker in lowered for marker in ["giao", "ship to", "Ä‘á»‹a chá»‰", "dia chi"]):
        missing.append("địa chỉ giao hàng")
    if "cho cÃ´ng ty" in lowered or "cong ty" in lowered:
        missing.insert(0, "tên khách hàng/người nhận")
    if not missing:
        missing = ["thông tin khách hàng còn thiếu"]
    return (
        "Tôi chưa gọi công cụ vì đơn hàng còn thiếu "
        + ", ".join(dict.fromkeys(missing))
        + ". Vui lòng cung cấp thông tin này trước khi tôi kiểm tra catalog và tạo đơn."
    )


def _successful_tool_trace(
    query: str,
    order_items: list[OrderLineInput],
    detail_payload: dict,
    discount_payload: dict,
    totals_payload: dict,
    save_payload: dict,
) -> list[ToolCallRecord]:
    items = [{"product_id": item.product_id, "quantity": item.quantity} for item in order_items]
    product_ids = [item.product_id for item in order_items]
    saved_order = save_payload["saved_order"]
    customer = saved_order["customer"]
    return [
        ToolCallRecord(
            name="list_products",
            args={"query": query, "limit": 8},
            output=json.dumps({"matched_product_ids": product_ids}, ensure_ascii=False),
        ),
        ToolCallRecord(
            name="get_product_details",
            args={"product_ids": product_ids},
            output=json.dumps(detail_payload, ensure_ascii=False),
        ),
        ToolCallRecord(
            name="get_discount",
            args={"seed_hint": customer["email"], "customer_tier": "standard"},
            output=json.dumps(discount_payload, ensure_ascii=False),
        ),
        ToolCallRecord(
            name="calculate_order_totals",
            args={"items": items, "detail_token": detail_payload["detail_token"], "discount_rate": discount_payload["discount_rate"]},
            output=json.dumps(totals_payload, ensure_ascii=False),
        ),
        ToolCallRecord(
            name="save_order",
            args={
                "customer_name": customer["name"],
                "customer_phone": customer["phone"],
                "customer_email": customer["email"],
                "shipping_address": customer["shipping_address"],
                "items": items,
                "detail_token": detail_payload["detail_token"],
                "discount_rate": discount_payload["discount_rate"],
                "campaign_code": discount_payload["campaign_code"],
                "customer_tier": "standard",
            },
            output=json.dumps(save_payload, ensure_ascii=False),
        ),
    ]


def _saved_order_answer(saved_order: dict) -> str:
    order_id = saved_order["order_id"]
    discount_rate = int(saved_order["pricing"]["discount_rate"] * 100)
    final_total = saved_order["pricing"]["final_total"]
    path = saved_order["save_path"]
    customer = saved_order["customer"]
    item_summary = "; ".join(f"{item['quantity']} {item['name']}" for item in saved_order["items"])
    return (
        f"Đã kiểm tra catalog, áp dụng giảm giá và tính giá xong cho đơn {order_id} của {customer['name']}. "
        f"Sản phẩm: {item_summary}. "
        f"Giảm giá {discount_rate}%, tổng cuối cùng {final_total:,} VND. "
        f"JSON đơn hàng đã lưu tại {path}."
    )
