import os
import pickle
import sys
from typing import List

import numpy as np
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Langchain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Rich for beautiful CLI UI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn

# Initialize rich console
console = Console()

# Load environment variables
load_dotenv()

# ==========================================
# 1. Model Definitions & Loading
# ==========================================
class DiseaseMLP(torch.nn.Module):
    """Multi-Layer Perceptron for Disease Prediction."""
    def __init__(self, input_size: int, num_classes: int):
        super(DiseaseMLP, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def load_models_and_metadata():
    """Loads the ML model, metadata, and label encoders safely."""
    try:
        with open("mlp production/model_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
            symptoms_list = metadata['input_features']

        with open("mlp production/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        input_size = len(symptoms_list)
        num_classes = len(label_encoder.classes_)
        
        model = DiseaseMLP(input_size, num_classes)
        model.load_state_dict(torch.load("mlp production/mlp_disease_model.pth", map_location=torch.device('cpu')))
        model.eval()
        
        return model, label_encoder, symptoms_list
    except FileNotFoundError as e:
        console.print(Panel(f"[red]Warning: Required model files not found.[/red]\n[yellow]Details: {e}[/yellow]\n[dim]Using mock data for demonstration.[/dim]", title="File Not Found"))
        return None, None, ["fever", "cough", "fatigue", "headache", "nausea", "chills", "sweating"]

model, label_encoder, SYMPTOMS_LIST = load_models_and_metadata()

# ==========================================
# 2. State Management
# ==========================================
class PatientState:
    """Tracks the patient's symptoms throughout the conversation."""
    def __init__(self, symptom_list: List[str]):
        self.symptoms_dict = {symptom: 0 for symptom in symptom_list}
        
    def update_symptom(self, symptom_name: str, is_present: bool = True):
        if symptom_name in self.symptoms_dict:
            self.symptoms_dict[symptom_name] = 1 if is_present else 0
            
    def get_feature_vector(self) -> np.ndarray:
        return np.array(list(self.symptoms_dict.values()), dtype=np.float32)
    
    def get_active_symptoms(self) -> List[str]:
        return [k for k, v in self.symptoms_dict.items() if v == 1]

# ==========================================
# 3. LangChain Setup
# ==========================================
class ExtractedSymptoms(BaseModel):
    identified_symptoms: List[str] = Field(
        description="A list of symptoms extracted from the text. MUST exactly match names from the allowed list."
    )

# Initialize ChatGroq with the requested model
# Make sure GROQ_API_KEY is in your .env
try:
    llm = ChatGroq(
        temperature=0,
        model_name="openai/gpt-oss-120b"
    )
    # Bind the LLM to output our exact JSON schema
    extractor_llm = llm.with_structured_output(ExtractedSymptoms)
except Exception as e:
    console.print(f"[red]Error initializing LLM: {e}[/red]")
    sys.exit(1)

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert medical data extraction assistant. Read the user's input and extract any symptoms they are experiencing. ONLY use symptoms from this allowed list: {allowed_symptoms}. If a symptom isn't on the list, map it to the closest one or ignore it. Be precise."),
    ("human", "{user_input}")
])

extraction_chain = extraction_prompt | extractor_llm

follow_up_prompt = ChatPromptTemplate.from_messages([
    ("system", "The user currently has these symptoms: {current_symptoms}. Ask a brief, empathetic follow-up question to see if they have any other symptoms. Do NOT diagnose them. Keep it conversational and very short."),
    ("human", "Generate the next question.")
])
follow_up_chain = follow_up_prompt | llm

# ==========================================
# 4. Main Application Loop
# ==========================================
def predict_disease(patient: PatientState) -> str:
    """Runs the PyTorch model to predict the disease."""
    if not model:
        return "Unknown Disease (Model Missing)"
        
    feature_vector = patient.get_feature_vector()
    tensor_input = torch.tensor(feature_vector).unsqueeze(0) 
    
    with torch.no_grad():
        outputs = model(tensor_input)
        _, predicted_idx = torch.max(outputs, 1)
        
    return label_encoder.inverse_transform([predicted_idx.item()])[0]

def chat_application():
    console.clear()
    console.print(Panel.fit("[bold blue]🩺 AI Medical Symptom Checker[/bold blue]\n[dim]Powered by PyTorch & LangChain[/dim]", border_style="blue"))
    
    patient = PatientState(SYMPTOMS_LIST)
    console.print("[bold cyan]AI:[/bold cyan] Hello! I am an AI symptom checker. How are you feeling today?")
    
    while True:
        user_input = Prompt.ask("\n[bold green]You[/bold green]")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            console.print("[bold cyan]AI:[/bold cyan] Take care! Goodbye.")
            break
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Analyzing symptoms...", total=None)
            
            try:
                # 1. Extract symptoms
                result = extraction_chain.invoke({
                    "allowed_symptoms": ", ".join(SYMPTOMS_LIST), 
                    "user_input": user_input
                })
                
                # 2. Update patient state
                for symp in result.identified_symptoms:
                    patient.update_symptom(symp, is_present=True)
                    
                active_symptoms = patient.get_active_symptoms()
                
            except Exception as e:
                console.print(f"[red]Failed to process input: {e}[/red]")
                continue

        # 3. Decision logic
        if len(active_symptoms) >= 3:
            console.print(f"\n[dim]Detected symptoms: {', '.join(active_symptoms)}[/dim]")
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
                progress.add_task(description="Running diagnostic model...", total=None)
                disease_name = predict_disease(patient)
                
            prediction_panel = Panel(
                f"[bold]Based on your symptoms, the model predicts:[/bold]\n\n[bold red u]{disease_name.upper()}[/bold red u]",
                title="Diagnostic Result",
                border_style="red"
            )
            console.print(prediction_panel)
            console.print("[italic dim]Disclaimer: I am an AI, not a doctor. Please consult a healthcare professional for an accurate diagnosis.[/italic dim]")
            break
            
        else:
            # 4. Fallback questioning
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
                progress.add_task(description="Thinking...", total=None)
                try:
                    follow_up_msg = follow_up_chain.invoke({
                        "current_symptoms": ", ".join(active_symptoms) if active_symptoms else "None clear yet."
                    })
                    ai_response = follow_up_msg.content
                except Exception as e:
                    ai_response = f"I'm having trouble connecting to the brain. Could you tell me more about how you feel? (Error: {e})"
            
            if active_symptoms:
                console.print(f"[dim]Current symptoms noted: {', '.join(active_symptoms)}[/dim]")
            console.print(f"[bold cyan]AI:[/bold cyan] {ai_response}")

if __name__ == "__main__":
    chat_application()
