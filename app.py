import streamlit as st
import torch
import torch.nn as nn
import json
import math

# Load vocabulary
with open("vocabulary.json", "r") as f:
    vocab = json.load(f)

# Page Config
st.set_page_config(page_title="Pseudocode to C++ Code Generator", layout="wide")

# Frontend Styling (Centered and Adjusted Width)
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #ffffff;
        color: #000000;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .stTextArea textarea, .stTextInput input, .stCode, .stButton button {
        background-color: #ffffff;
        color: #000000;
        border-radius: 5px;
        border: 1px solid #cccccc;
        width: 100% !important;  /* Adjust width here */
        margin: 0 auto;  /* Center align */
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 10px auto;  /* Center align */
        cursor: pointer;
        border-radius: 5px;
        width: auto;  /* Auto width for button */
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stTextArea textarea {
        height: 200px;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
    }
    .stTitle {
        text-align: center;
        width: 100%;
    }
    .stMarkdown {
        text-align: center;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Transformer Configuration
class Config:
    vocab_size = 12388
    max_length = 100
    embed_dim = 256
    num_heads = 8
    num_layers = 2
    feedforward_dim = 512
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# Transformer Model
class Seq2SeqTransformer(nn.Module):
    def __init__(self, config):
        super(Seq2SeqTransformer, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_encoding = PositionalEncoding(config.embed_dim, config.max_length)
        self.transformer = nn.Transformer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout
        )
        self.fc_out = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src) * math.sqrt(config.embed_dim)
        tgt_emb = self.embedding(tgt) * math.sqrt(config.embed_dim)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)
        out = self.transformer(src_emb.permute(1, 0, 2), tgt_emb.permute(1, 0, 2))
        out = self.fc_out(out.permute(1, 0, 2))
        return out

# Load Models
@st.cache_resource
def load_model(path):
    model = Seq2SeqTransformer(config).to(config.device)
    model.load_state_dict(torch.load(path, map_location=config.device))
    model.eval()
    return model

pseudo_to_cpp_model = load_model("transformer_coder.pth")

# Translation Function
def translate(model, input_tokens, vocab, device, max_length=50):
    model.eval()
    input_ids = [vocab.get(token, vocab["<unk>"]) for token in input_tokens]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    output_ids = [vocab["<start>"]]

    for _ in range(max_length):
        output_tensor = torch.tensor(output_ids, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = model(input_tensor, output_tensor)
        next_token_id = predictions.argmax(dim=-1)[:, -1].item()
        
        if next_token_id == vocab["<end>"]:
            break  # Stop generating when <end> token is reached
        
        output_ids.append(next_token_id)

    id_to_token = {idx: token for token, idx in vocab.items()}
    generated_tokens = [id_to_token.get(idx, "<unk>") for idx in output_ids[1:]]
    
    return " ".join(generated_tokens)  # Return translated code without <end> token

# Streamlit UI (Centered and Adjusted Width)
st.title("Pseudocode to C++ Code Generator")

# Input text area for pseudocode
pseudocode_input = st.text_area("Enter your pseudocode here:", height=200)

# Button to generate code
if st.button("Generate C++ Code"):
    if pseudocode_input:
        with st.spinner("Generating C++ code..."):
            try:
                tokens = pseudocode_input.strip().split()
                generated_code = translate(pseudo_to_cpp_model, tokens, vocab, config.device)
                st.text_area("Generated C++ Code:", generated_code, height=300)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some pseudocode to generate C++ code.")