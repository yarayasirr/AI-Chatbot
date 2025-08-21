import random
import json
import torch
import customtkinter as ctk
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# -------------------------
# Load intents + trained model
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("intents.json", "r") as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"

def chatbot_response(user_input):
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).reshape(1, X.shape[0])
    X = X.to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "Sorry, I didn’t quite get that."

# -------------------------
# GUI with CustomTkinter
# -------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.title("AI Chatbot")
root.geometry("600x600")

# 🎨 Teal theme colors
BG_COLOR = "#012A36"   # deep teal background
BOT_COLOR = "#028090"  # bright teal for bot replies
USER_COLOR = "#05668D" # darker teal for user
TEXT_COLOR = "white"

# Chat frame
chat_frame = ctk.CTkScrollableFrame(root, width=560, height=480, fg_color=BG_COLOR)
chat_frame.pack(pady=10, padx=10, fill="both", expand=True)

# Input frame
input_frame = ctk.CTkFrame(root, fg_color=BG_COLOR)
input_frame.pack(pady=10, padx=10, fill="x")

user_entry = ctk.CTkEntry(
    input_frame,
    placeholder_text="Type your message...",
    width=400,
    fg_color="#033649",  # muted teal
    text_color=TEXT_COLOR
)
user_entry.pack(side="left", padx=10, pady=10)

def add_message(sender, message, color, anchor="w"):
    label = ctk.CTkLabel(
        chat_frame,
        text=f"{sender}: {message}",
        text_color=TEXT_COLOR,
        fg_color=color,
        corner_radius=15,
        wraplength=400,
        anchor="w"
    )
    label.pack(anchor=anchor, pady=5, padx=10, fill="x")

def send_message(event=None):
    user_input = user_entry.get()
    if user_input.strip():
        add_message("You", user_input, USER_COLOR, "e")
        user_entry.delete(0, "end")

        bot_reply = chatbot_response(user_input)
        add_message(bot_name, bot_reply, BOT_COLOR, "w")

send_btn = ctk.CTkButton(
    input_frame,
    text="Send",
    command=send_message,
    fg_color=BOT_COLOR,
    hover_color="#00A896"
)
send_btn.pack(side="right", padx=10)

# Bind Enter key
user_entry.bind("<Return>", send_message)

root.mainloop()
