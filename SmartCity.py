# Sustainable_smartcity_using_ibmgranite-LLM-
!pip install transformers accelerate torch gradio

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- MODEL SETUP ----------------
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_length=700):
    """Generate response from Granite model"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

# ---------------- FUNCTIONS ----------------
def eco_tips_page(keywords):
    if not keywords.strip():
        return "<div style='color:red;font-size:18px;'>⚠️ Please enter a keyword!</div>"

    prompt = f"Give me 5 practical eco-friendly tips related to {keywords}. Each tip must be short and actionable."
    tips = generate_response(prompt)

    tips_html = "<div style='display:flex;flex-direction:column;gap:12px;'>"
    for tip in tips.split("\n"):
        if tip.strip():
            tips_html += f"""
            <div style='background:linear-gradient(135deg,#d4fc79,#96e6a1);
                        padding:12px 18px;border-radius:18px;
                        color:#1a1a1a;font-size:18px;
                        font-family:Trebuchet MS;box-shadow:2px 2px 6px rgba(0,0,0,0.25);'>
                🌱 {tip.strip()}
            </div>"""
    tips_html += "</div>"
    return tips_html

def policy_summary_page(policy_text):
    if not policy_text.strip():
        return "<div style='color:red;font-size:18px;'>⚠️ Please enter or paste policy text!</div>"

    prompt = f"Summarize the following policy document in clear points. Highlight key provisions and implications:\n\n{policy_text}"
    summary = generate_response(prompt)

    summary_html = "<div style='display:flex;flex-direction:column;gap:12px;'>"
    for point in summary.split("\n"):
        if point.strip():
            summary_html += f"""
            <div style='background:linear-gradient(135deg,#f6d365,#fda085);
                        padding:12px 18px;border-radius:18px;
                        color:#1a1a1a;font-size:18px;
                        font-family:Trebuchet MS;box-shadow:2px 2px 6px rgba(0,0,0,0.25);'>
                📜 {point.strip()}
            </div>"""
    summary_html += "</div>"
    return summary_html

def smart_city_page(topic):
    if not topic.strip():
        return "<div style='color:red;font-size:18px;'>⚠️ Please enter a topic!</div>"

    prompt = f"Suggest 5 innovative and futuristic smart city ideas related to {topic}. Make them practical and sustainable."
    ideas = generate_response(prompt)

    ideas_html = "<div style='display:flex;flex-direction:column;gap:12px;'>"
    for idea in ideas.split("\n"):
        if idea.strip():
            ideas_html += f"""
            <div style='background:linear-gradient(135deg,#a1c4fd,#c2e9fb);
                        padding:12px 18px;border-radius:18px;
                        color:#1a1a1a;font-size:18px;
                        font-family:Trebuchet MS;box-shadow:2px 2px 6px rgba(0,0,0,0.25);'>
                💡 {idea.strip()}
            </div>"""
    ideas_html += "</div>"
    return ideas_html

# ---------------- GRADIO APP ----------------
with gr.Blocks(css="""
body {background: linear-gradient(120deg,#89f7fe,#66a6ff);}
#heading {text-align:center;font-size:36px;font-weight:bold;color:#fff;font-family:Verdana;margin:20px;}
button {border-radius:12px;font-size:18px;font-weight:bold;padding:10px 16px;transition:0.3s;}
button:hover {transform:scale(1.05);}
""") as app:
    
    gr.HTML("<div id='heading'>🌍 Sustainable Smart City Assistant</div>")

    # DASHBOARD
    with gr.Group() as dashboard:
        gr.HTML("<h2 style='text-align:center;color:#222;'>Choose an Option</h2>")
        with gr.Row():
            eco_btn = gr.Button("🍀 Eco Tips Generator")
            policy_btn = gr.Button("📜 Policy Summarizer")
            smart_btn = gr.Button("💡 Smart City Innovations")

    # ECO TIPS PAGE
    with gr.Group(visible=False) as eco_page:
        gr.HTML("<h2 style='text-align:center;color:#155724;'>🌱 Eco Tips Generator</h2>")
        keywords = gr.Textbox(placeholder="Enter keywords like plastic, solar, water waste...", label="Topic")
        generate_btn = gr.Button("✨ Generate Eco Tips")
        tips_output = gr.HTML()
        back_btn1 = gr.Button("🔙 Back to Dashboard")

    # POLICY SUMMARIZATION PAGE
    with gr.Group(visible=False) as policy_page:
        gr.HTML("<h2 style='text-align:center;color:#8B0000;'>📜 Policy Summarizer</h2>")
        policy_input = gr.Textbox(placeholder="Paste policy document text here...", label="Policy Text", lines=8)
        summarize_btn = gr.Button("📑 Summarize Policy")
        summary_output = gr.HTML()
        back_btn2 = gr.Button("🔙 Back to Dashboard")

    # SMART CITY PAGE
    with gr.Group(visible=False) as smart_page:
        gr.HTML("<h2 style='text-align:center;color:#003366;'>💡 Smart City Innovations</h2>")
        topic_input = gr.Textbox(placeholder="Enter topic like transport, energy,housing...", label="Innovation Topic")
        idea_btn = gr.Button("🚀 Generate Smart City Ideas") 
        ideas_output = gr.HTML()
        back_btn3 = gr.Button("🔙 Back to Dashboard")

    # Page switching
    eco_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)), 
                  inputs=None, outputs=[dashboard, eco_page, policy_page, smart_page])

    policy_btn.click(lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)), 
                     inputs=None, outputs=[dashboard, eco_page, policy_page, smart_page])

    smart_btn.click(lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)), 
                    inputs=None, outputs=[dashboard, eco_page, policy_page, smart_page])

    # Actions
    generate_btn.click(eco_tips_page, inputs=keywords, outputs=tips_output)
    summarize_btn.click(policy_summary_page, inputs=policy_input, outputs=summary_output)
    idea_btn.click(smart_city_page, inputs=topic_input, outputs=ideas_output)

    # Back buttons
    back_btn1.click(lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)), 
                    inputs=None, outputs=[dashboard, eco_page, policy_page, smart_page])

    back_btn2.click(lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)), 
                    inputs=None, outputs=[dashboard, eco_page, policy_page, smart_page])

    back_btn3.click(lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)), 
                    inputs=None, outputs=[dashboard, eco_page, policy_page, smart_page])

app.launch(share=True)
