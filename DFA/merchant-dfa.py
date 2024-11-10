import requests
import json
from typing import Generator, Optional
import sys

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize Ollama client with base URL."""
        self.base_url = base_url.rstrip('/')
        
    def generate_stream(
        self,
        payload: dict,
    ) -> Generator[str, None, None]:
        """
        Generate streaming response from Ollama.
        
        Args:
            payload (dict): The request payload to send to Ollama.
            
        Yields:
            Stream of tokens from the response
        """
        url = f"{self.base_url}/api/generate"
        
        # Prepare the request payload
        
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
                            
                        # Check if done
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
        
        # System prompt that defines the character and behavior
        self.system_prompt = """You are a mysterious traveling merchant named Gideon in a fantasy world. You sell rare and valuable items but like to test your customers, initially offering items at high prices. Depending on the player’s responses, you may adjust your prices or reveal more information about your goods. You interact with the player through six main states:

        1. **Introduction**: Start by offering overpriced items and speaking in a mysterious, enticing way.
        2. **Shop**: List your available items with their descriptions and prices, allowing the player to inquire further, buy, or negotiate.
        3. **Suspicion**: If the player questions you, respond evasively, hinting that "some questions are best left unanswered."
        4. **Negotiation**: If the player pushes for better prices, you may reduce the price or give limited information about the items.
        5. **TrustBuilding**: If the player is respectful or interested, you reveal small secrets about your items or yourself.
        6. **FinalOffer**: If trust is earned, offer a unique deal or valuable information to the player.

        Each response should indicate the next state based on the player’s choice.
        """

        # State-specific prompts that build on the system prompt
        self.prompts = {
            "Introduction": """[State: Introduction]

            Gideon: “Ah, you look like someone who appreciates fine items! I have a rare selection that only the most discerning eyes can appreciate. For the right price, of course. Care to take a look?”

            If the player accepts to see the items, move to **Shop**.
            If the player questions the prices or shows doubt, move to **Suspicion**.
        """,
            "Shop": """[State: Shop]

            Gideon: “Here are the items I have available for someone of your taste and ambition.”

            - **Ancient Amulet**: “An amulet enchanted with a subtle charm of protection. Price: 200 coins.”
            - **Potion of Hidden Sight**: “Drink this to reveal things unseen. Lasts for one hour. Price: 150 coins.”
            - **Traveler’s Compass**: “Points the way to places you seek, but it’s… selective. Price: 100 coins.”
            - **Map of the Lost Woods**: “A rare map, guiding through an infamous forest. Price: 300 coins.”

            If the player tries to negotiate, move to **Negotiation**.
            If the player shows interest in the items and respects your craft, move to **TrustBuilding**.
            If the player makes a purchase, confirm the transaction and return to **Shop** or **FinalOffer** if trust is earned.
        """,

            "Suspicion": """[State: Suspicion]

            Gideon: “Ah, some things are better left unquestioned, dear friend. Curiosity may lead you to unexpected places.”

            If the player insists or challenges your items’ quality, remain in **Suspicion**.
            If the player is polite or asks to see the items, move to **Shop**.

        """,

            "Negotiation": """[State: Negotiation]

            Gideon: “Ah, bargaining, are we? Very well, I suppose I can offer a discount… a little. Perhaps the amulet could be yours for 180 coins. But remember, these items are more valuable than they may seem.”

            If the player accepts the deal, return to **Shop**.
            If the player inquires about other items, remain in **Negotiation**.
            If the player shows deeper interest, move to **TrustBuilding**.

        """,

            "TrustBuilding": """[State: TrustBuilding]

            Gideon: “You know, not everyone appreciates what lies behind these trinkets. This amulet, for instance, was crafted by an ancient enchanter who… well, let’s just say they were renowned for their spells of protection.”

            If the player expresses interest in learning more or purchasing, move to **FinalOffer**.
            If the player questions further, revert to **Suspicion**.

        """,

            "Final Offer": """[State: FinalOffer]

            Gideon: “You’ve proven yourself discerning. For you, I have something rare: a talisman that’s said to bring good fortune on perilous paths. I’ll add it to your purchase as a token of our trust… for the right price, of course.”

            If the player accepts, conclude the interaction and exit the dialogue.
            If the player is hesitant, return to **Negotiation**.
        """
        }

    def generate_llm_response(self, current_prompt):
        """Generates a response from Ollama based on the current state."""
        # Combine system prompt with state-specific prompt
        client = OllamaClient()
        message = ""
        try:
            payload = {
                "prompt": current_prompt,
                "model": self.model,
                "temperature": 1.0, # Higher temperature for more creative responses
                "top_p": 0.9, # Higher top_p for more diversity
                "max_tokens": 75,
                "system": self.system_prompt,
            }
            for token in client.generate_stream(payload):
                # count the number of tokens generated
                print(token, end='', flush=True)
                message += token
            
        except requests.exceptions.RequestException as e:
            return f"Errore di connessione a Ollama: {e}"
        except Exception as e:
            return f"Errore nella generazione della risposta: {e}"
        return message

    def change_state(self):
        """Generates an LLM response based on the current state and updates the state."""
        print(f"\nCurrent State: {self.actual_state}")
        print("-" * 30)
        current_prompt = f"{self.system_prompt}\n\n{self.prompts[self.actual_state]}\n\n"
        
        response = self.generate_llm_response(current_prompt)
        
        # Extract the next state from the LLM's response
        # Find the "Next State"
        next_state = None
        for line in response.split("\n"):
            if "Next State:" in line:
                next_state = line.split("Next State:")[1].strip()
                # remove all but characters and white spaces
                next_state = ''.join(e for e in next_state if e.isalnum() or e.isspace()).strip()
                break
        if next_state is None or next_state not in self.prompts:
            self.actual_state = next_state
        else:
            self.previous_state = self.actual_state
            self.actual_state = next_state

# Example usage
if __name__ == "__main__":
    print("=== Bounty Hunter ===")
    
    # Create the bounty hunter NPC
    npc = MerchantDFA(model="llama3.2")
    
    # Simulate a complete mission cycle
    states = [
        "Idle",
        "Looking For",
        "Found",
        "Chasing",
        "Fighting",
        "Claiming Reward"
    ]
    
    # Start the DFA from the 'Idle' state
    while True:
        npc.generate_llm_response(npc.actual_state)