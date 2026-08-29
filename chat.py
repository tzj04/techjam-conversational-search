from starter.agent import Agent

agent = Agent("data/catalog.jsonl")

session_id = "manual"
profile = {
    "summary": "Prefers comfortable, durable clothing",
    "preference_tags": ["fit", "comfort", "durability"],
}
agent.reset(session_id, profile)
print("Agent ready. Type your message ('quit' or empty line to exit).\n")

message = input("You: ")
for turn in range(1, 11):
    response = agent.respond(session_id, message, turn, 10)
    print(f"\nAgent (turn {turn}): {response['message']}")
    print(f"  ask_attribute: {response['ask_attribute']}")
    for i, rec in enumerate(response["recommendations"], 1):
        product = agent._products.get(rec["parent_asin"], {})
        title = str(product.get("title") or rec["parent_asin"])
        if len(title) > 90:
            title = title[:87] + "..."
        price = product.get("price")
        price_str = f"${price:.2f}" if isinstance(price, (int, float)) else "n/a"
        rating = product.get("average_rating") or 0.0
        rating_str = f"{rating:.1f}*" if rating else "unrated"
        store = str(product.get("store") or "")
        print(f"  {i:2}. {title}")
        print(f"      {price_str} | {rating_str} | {store} | {rec['parent_asin']}")
    if turn < 10:
        print()
        message = input("You: ")
        if not message.strip() or message.strip().lower() == "quit":
            break
