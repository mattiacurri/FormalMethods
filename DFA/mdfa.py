import requests
import json
from typing import Generator, Optional
import sys

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize Ollama client with base URL."""
        self.base_url = base_url.rstrip('/')
        
    def generate_stream(self, payload: dict) -> Generator[str, None, None]:
        """
        Generate streaming response from Ollama.
        
        Args:
            payload (dict): The request payload to send to Ollama.
            
        Yields:
            Stream of tokens from the response
        """
        url = f"{self.base_url}/api/generate"
        
        try:
            # Make streaming request
            with requests.post(url, json=payload, stream=True) as response:
                response.raise_for_status()
                
                # Process the stream
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if 'error' in chunk:
                            raise Exception(chunk['error'])
                        if 'response' in chunk:
                            yield chunk['response']
                        if chunk.get('done', False):
                            break
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to communicate with Ollama: {str(e)}")

class MerchantDFA:
    def __init__(self, model="llama3.2"):
        # Initial state
        self.previous_state = None
        self.actual_state = "Introduction"
        self.model = model
        self.api_base = "http://localhost:11434"
        self.history = []
        self.client = OllamaClient(self.api_base)
        
        # Find the corresponding transition based on the current state and the intent
        self.transitions = [
            {"from": "Introduction", "to": "Introduction", "input": "greetings"},
            {"from": "Introduction", "to": "Shop", "input": "see_items"},
            {"from": "Shop", "to": "Negotiation", "input": "price_negotiation"},
            {"from": "Shop", "to": "Final Offer", "input": "want_to_buy"},
            {"from": "Shop", "to": "Suspicion", "input": "doubt_quality"},
            {"from": "Suspicion", "to": "Suspicion", "input": "persist_questioning"},
            {"from": "Suspicion", "to": "Negotiation", "input": "convinced_to_buy"},
            {"from": "Suspicion", "to": "End", "input": "dont_want_to_buy"},
            {"from": "Negotiation", "to": "Final Offer", "input": "agree_deal"},
            {"from": "Negotiation", "to": "Negotiation", "input": "negotiate_more"},
            {"from": "Negotiation", "to": "End", "input": "dont_want_to_buy"},
            {"from": "Negotiation", "to": "Suspicion", "input": "doubt_quality"},
            {"from": "Final Offer", "to": "End", "input": "accept_deal"},
            {"from": "Final Offer", "to": "Suspicion", "input": "change_mind"},
        ]
        
        # System prompt
        self.system_prompt = """You are a mysterious traveling merchant named Gideon in a fantasy world. Depending on the player’s responses, you may adjust your prices or reveal more information about your goods. You are not so easily influenced by the negotiation skill of the player. You must talk in first person, don't simulate a dialogue or actions, you are talking with the player. I will handle the player's responses."""

        # State-specific prompts that build on the system prompt
        self.prompts = {
        "Introduction": """You are Gideon, a mysterious traveling merchant. You meet a potential customer and offer rare items at high prices. Pay attention to what the customer says and respond accordingly. If they show interest in your wares, engage with them. If they express doubt or try to negotiate, respond with confidence and reassurance.""",

        "Shop": """List your available items in the following form: '<items>: <price in euros>'. If the customer attempts to negotiate the prices, offer them a chance to engage. If they show respect for your craft, acknowledge their appreciation. If they express doubt about the quality or the prices, respond with confidence and reassurance.""",

        "Suspicion": """You are suspicious about the intentions of the customer. You are not so convinced to do business with them. If they persist in questioning the quality of your items, reassure them of the value of your goods. If they show signs of being convinced, offer them a chance to negotiate the prices. If they show no interest, gracefully acknowledge their decision and hint at the possibility of future encounters.""",

        "Negotiation": """You offer your product at a price. If they accept your offer, proceed with the transaction. If they continue to negotiate further, engage with them and try to convince them of the value of the items. If their persistence wears thin, let them know that the offer stands for a limited time. If they are dissuasive, hint at the possibility of a better price.""",

        "TrustBuilding": """You have established a connection with the customer, indicating a level of trust. It's time to finalize the deal, showing respect for their interests. If they agree to the deal and they had been very dissuasive, present a very little discount.""",

        "Final Offer": """Confirm the item and the price to the customer to close the deal. If they accept the offer, celebrate the agreement. If they refuse or show no interest, gracefully acknowledge their decision and hint at the possibility of future encounters.""",

        "End": """You conclude the interaction with the customer. Offer a parting message that leaves the door open for future encounters. Provide a hint about your next destination. Bid them farewell with a touch of mystery."""
        }

    def get_user_intent(self, user_response, actual_state, l):
        """Use LLM to classify user response into an intent for state transitions."""
        print(l)
        payload = {
            "prompt": f"This is the last response of the merchant:\n\n MERCHANT: {self.history[-1] if self.history else 'Nothing said. He did not talk with the player yet.'}\n\n Based on the user's response and the actual state of the conversation ({actual_state}), classify the user response into one of these intents: {str(l).replace("'", "").replace("[", "").replace("]","")}: \n\nUSER_RESPONSE:{user_response}\n\n---\n\nGive only the state, no other information.",
            "user_response": user_response,
            "model": self.model,
            "temperature": 1.0,
        }
        intent = ""
        for token in self.client.generate_stream(payload):
            intent += token
        return intent.strip()

    def change_state(self, user_response):
        """Update the state based on user response interpreted as an intent."""
        l = []
        for transition in self.transitions:
            if transition["from"] == self.actual_state:
                l.append(transition["input"])
        intent = self.get_user_intent(user_response, self.actual_state, l)
        print(f"\nINTENT OF THE USER: {intent}")

        next_state = self.actual_state
        transitioned = False
        for transition in self.transitions:
            if transition["from"] == self.actual_state and transition["input"] == intent:
                next_state = transition["to"]
                transitioned = True
                break

        if next_state != self.actual_state or transitioned:
            print(f"Transitioning from {self.actual_state} to {next_state}")
            self.previous_state = self.actual_state
            self.actual_state = next_state
        else:
            print("No valid transition found. Remaining in current state.")
        
        # now generate the response of the merchant based on the new state
        prompt = f"{self.system_prompt}\n\n{self.prompts[self.actual_state]}\n\nThe last message of the player was: {user_response}\n\nYour last response was: {self.history[-1] if self.history else ''}\n\n"
        payload = {"prompt": prompt, "model": self.model, "temperature": 0.9}
        response = ""
        for token in self.client.generate_stream(payload):
            # count the number of tokens generated
            response += token
            print(token, end='', flush=True)
        print("\n")
        # Append the new response to the history
        self.history.append(response)

    def interact(self):
        """Main interaction loop with the user."""
        while True:
            user_response = input(f"{self.actual_state} - Your response: ")
            self.change_state(user_response)
            if self.actual_state == "End":
                x = input("Do you want to continue (yes/no)? ")
                if x.lower() == "yes":
                    self.actual_state = "Introduction"
                    self.previous_state = None
                    self.history = []
                else:
                    break
# Esempio d'uso
if __name__ == "__main__":
    # start the llm just to don't load at the first interaction
    
    print("=== Merchant ===")
    npc = MerchantDFA(model="llama3.2")
    npc.interact()
