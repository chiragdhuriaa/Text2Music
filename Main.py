import tkinter as tk
import customtkinter as ctk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = tk.Tk()
app.geometry("532x632")
app.title("Text2Music Synthesizer")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, text="Generated Music will save here", height=40, width=512, font=("Arial", 20), text_color="white")
lmain.place(x=10, y=110)

tokenizer = AutoTokenizer.from_pretrained('sander-wood/text-to-music')
model = AutoModelForSeq2SeqLM.from_pretrained('sander-wood/text-to-music')
max_length = 1024
top_p = 0.9
temperature = 1.0

def top_p_sampling(probs, top_p, return_probs=False):
    sorted_indices = torch.argsort(probs, descending=True)
    cumulative_probs = torch.cumsum(probs[sorted_indices], dim=0)
    sorted_indices_to_remove = cumulative_probs > top_p
    if sorted_indices_to_remove[0].item() == True:
        sorted_indices_to_remove[0] = False
    probs[sorted_indices[sorted_indices_to_remove]] = 0
    probs = probs / probs.sum()
    if return_probs:
        return probs
    else:
        return torch.multinomial(probs, 1)

def temperature_sampling(probs, temperature):
    if temperature == 0:
        return torch.argmax(probs, dim=-1)
    else:
        probs = torch.pow(probs, 1.0 / temperature)
        return torch.multinomial(probs, 1).item()

def generate_music():
    prompt_text = prompt.get()
    
    input_ids = tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=max_length)['input_ids']
    decoder_start_token_id = model.config.decoder_start_token_id
    eos_token_id = model.config.eos_token_id
    decoder_input_ids = torch.tensor([[decoder_start_token_id]])

    tune = "X:1\n"
    
    for t_idx in range(max_length):
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        probs = outputs.logits[0][-1]
        probs = torch.nn.Softmax(dim=-1)(probs).detach()
        sampled_id = temperature_sampling(probs=top_p_sampling(probs, top_p=top_p, return_probs=True), temperature=temperature)
        decoder_input_ids = torch.cat((decoder_input_ids, torch.tensor([[sampled_id]])), 1)
        if sampled_id != eos_token_id:
            continue
        else:
            tune += tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
            with open("generated_music.abc", "w") as f:
                f.write(tune)
            lmain.configure(text="Music saved as 'generated_music.abc'")
            print(tune)
            break

trigger = ctk.CTkButton(app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate_music)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
