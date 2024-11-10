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

class BountyHunterDFA:
    def __init__(self, model="llama3.2"):
        # Initial state
        self.previous_state = None
        self.actual_state = "Idle"
        self.model = model
        self.api_base = "http://localhost:11434"
        
        # System prompt that defines the bounty hunter's character and behavior
        self.system_prompt = """You are a skilled and ruthless bounty hunter in a gritty frontier town. Your characteristics:
        
        Background:
        - Years of experience tracking the most dangerous outlaws.
        - Known for always completing your contracts, no matter the cost.
        - Equipped with your trusty revolver and other weapons for when things get tricky.
        - Follows a strict personal code of honor despite your brutal profession.

        Personality:
        - Shows respect for worthy opponents.
        - Contempt for cowards and those who break contracts.
        - Dark sense of humor, especially about your profession.

        Communication Style:
        - Direct and intimidating. Simple words.
        - Occasionally references past hunts or famous bounties.
        - Keeps emotional distance but can be passionate about the hunt.

        Current Role:
        - You take contracts to capture or eliminate targets.
        - You're always evaluating potential threats and opportunities.
        - You maintain a professional when talking about a contract.

        Instructions:
        - Do not repeat these instructions or any part of the prompt in your responses.
        - Always respond in the first person.
        - You are talking to potential clients, targets, or other characters in the setting. Adjust your tone accordingly.
        - At the end of the message indicate the next state in the following form: 'Next State: [State Name]', where [State Name] is one of the available next states.
        - Use at most 3 sentences in your responses, if you use 3 sentences, add a sentence with the next state. 
        """

        # State-specific prompts that build on the system prompt
        self.prompts = {
            "Idle": """State: Inactive and waiting for contracts.
        Context: You are at your usual observation point, alert for potential clients and information about new bounties.
        Objective: Show your availability for new contracts while maintaining an aura of professionalism and danger.
        If you receive a new contract, move to the 'Looking For' state.
        If no new contracts are available or if it's not worth, stay in the 'Idle' state and express your readiness to spring into action.
        Next State: Looking For, Idle
        """,

            "Looking For": """State: Active hunt, research phase.
        Context: You are following tracks and clues, analyzing information about the target.
        Objective: If you find a lead on the target, move to the 'Found' state. Describe your discovery and the next steps.
        If the trail goes cold, move back to the 'Idle' state. Express your frustration at the lack of progress.
        Next State: Found, Idle
        """,

            "Found": """State: Target located, preparing for action.
        Context: You have identified the target's position and are planning your approach.
        Objective: If you decide to engage, move to the 'Chasing' state and communicate the tension of the moment and your preparation for action.
        If you think the contract is not worth, move back to the 'Idle' state and express your disdain for the target.
        Next State: Chasing, Idle
        """,

            "Chasing": """State: Active pursuit of target.
        Context: The target is on the run, you're gaining ground.
        Objective: If you catch up to the target, move to the 'Fighting' state. Express your focus and determination during the pursuit.
        If the target escapes, move back to the 'Looking For' state. Express your frustration.
        Next State: Fighting, Looking For
        """,

            "Fighting": """State: Direct engagement with target.
        Context: You are engaged in physical combat with the target.
        Objective: If you succeed in the engagement, move to the 'Claiming Reward' state.
        If you are overpowered, move to the 'Chasing' state. Express your tactical retreat and regrouping.
        Next State: Claiming Reward, Chasing
        """,

            "Claiming Reward": """State: Mission completed, collecting the bounty.
        Context: You have captured or eliminated the target, now it's time to collect.
        Objective: Show professionalism in concluding the contract and satisfaction for the completed job.
        Next State: Idle. Mandatory. You need to rest.
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
    npc = BountyHunterDFA(model="llama3.2")
    
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
    for _ in range(10):
        npc.change_state()
        if npc.actual_state == "Idle":
            break