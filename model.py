import os
# Fix OpenMP conflict before importing other libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import faiss
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import torch
from openai import OpenAI
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeTranslator:
    """
    A class for translating code between Java and C# using semantic search and LLM generation.
    """
    
    def __init__(self, 
                 dataset_path: str = "./dataset",
                 api_key: Optional[str] = None,
                 model_name: str = "llama-3.3-70b-versatile"):
        """
        Initialize the CodeTranslator.
        
        Args:
            dataset_path: Path to the dataset directory
            api_key: Groq API key (if not provided, will try to get from environment)
            model_name: Name of the model to use for translation
        """
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dimension = 768
        
        # Initialize API client
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        
        # Load components
        self._load_mappings()
        self._load_vectors()
        self._build_indices()
        self._load_embedding_model()
        
        logger.info("CodeTranslator initialized successfully")
    
    def _load_mappings(self) -> None:
        """Load the retrieval mappings for Java and C# code snippets."""
        try:
            mapping_path = self.dataset_path / "retrieval_mapping.pkl"
            with open(mapping_path, "rb") as f:
                self.retrieval_mapping = pickle.load(f)
            
            self.java_snippets = self.retrieval_mapping["java"]
            self.cs_snippets = self.retrieval_mapping["cs"]
            
            logger.info(f"Loaded {len(self.java_snippets)} Java and {len(self.cs_snippets)} C# snippets")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Mapping file not found at {mapping_path}")
        except Exception as e:
            raise Exception(f"Error loading mappings: {e}")
    
    def _load_vectors(self) -> None:
        """Load the pre-computed embedding vectors."""
        try:
            self.cs_vectors = np.load(self.dataset_path / "cs_vectors.npy", allow_pickle=True)
            self.java_vectors = np.load(self.dataset_path / "java_vectors.npy", allow_pickle=True)
            
            logger.info(f"Loaded vectors: Java({self.java_vectors.shape}), C#({self.cs_vectors.shape})")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Vector files not found: {e}")
        except Exception as e:
            raise Exception(f"Error loading vectors: {e}")
    
    def _build_indices(self) -> None:
        """Build FAISS indices for similarity search."""
        try:
            self.java_index = faiss.IndexFlatL2(self.dimension)
            self.cs_index = faiss.IndexFlatL2(self.dimension)
            
            self.java_index.add(self.java_vectors.astype("float32"))
            self.cs_index.add(self.cs_vectors.astype("float32"))
            
            logger.info("FAISS indices built successfully")
            
        except Exception as e:
            raise Exception(f"Error building FAISS indices: {e}")
    
    def _load_embedding_model(self) -> None:
        """Load the CodeBERT model for generating embeddings."""
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
            self.model = RobertaModel.from_pretrained("microsoft/codebert-base")
            self.model.to(self.device)
            self.model.eval()
            
            # Disable gradients for inference to save memory
            for param in self.model.parameters():
                param.requires_grad = False
            
            logger.info(f"CodeBERT model loaded on {self.device}")
            
        except Exception as e:
            raise Exception(f"Error loading embedding model: {e}")
    
    def get_embedding(self, code_snippet: str) -> np.ndarray:
        """
        Generate embedding for a code snippet.
        
        Args:
            code_snippet: The code to embed
            
        Returns:
            numpy array of the embedding
        """
        try:
            # Tokenize and encode
            inputs = self.tokenizer(
                code_snippet, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,  # Increased for longer snippets
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
                return embedding.squeeze().cpu().numpy()
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise Exception(f"Failed to generate embedding: {e}")
    
    def search_similar_examples(self, 
                               query: str, 
                               direction: str, 
                               top_k: int = 3) -> List[Tuple[str, str]]:
        """
        Search for similar code examples using FAISS.
        
        Args:
            query: The code to find similar examples for
            direction: Translation direction ("java-to-cs" or "cs-to-java")
            top_k: Number of similar examples to retrieve
            
        Returns:
            List of (source_code, target_code) tuples
        """
        try:
            query_embedding = self.get_embedding(query).astype("float32").reshape(1, -1)
            
            if direction == "java-to-cs":
                # Search in Java index for similar Java code
                D, I = self.java_index.search(query_embedding, top_k)
                similar_examples = [(self.java_snippets[i], self.cs_snippets[i]) for i in I[0]]
            elif direction == "cs-to-java":
                # Search in C# index for similar C# code
                D, I = self.cs_index.search(query_embedding, top_k)
                similar_examples = [(self.cs_snippets[i], self.java_snippets[i]) for i in I[0]]
            else:
                raise ValueError(f"Invalid direction: {direction}")
            
            logger.info(f"Found {len(similar_examples)} similar examples")
            return similar_examples
            
        except Exception as e:
            logger.error(f"Error searching similar examples: {e}")
            raise Exception(f"Failed to search similar examples: {e}")
    
    def format_prompt(self, 
                     query: str, 
                     direction: str, 
                     similar_examples: List[Tuple[str, str]]) -> str:
        """
        Format the prompt for the LLM.
        
        Args:
            query: The code to translate
            direction: Translation direction
            similar_examples: List of similar code pairs
            
        Returns:
            Formatted prompt string
        """
        if direction == "java-to-cs":
            source_lang, target_lang = "Java", "C#"
            template = """You are an expert code translator. Translate the given Java code to idiomatic C# code.

Rules:
- Return ONLY the translated C# code without explanations
- Maintain the same functionality and logic
- Use C# naming conventions (PascalCase for methods, properties)
- Use appropriate C# syntax and idioms
- Handle Java-specific constructs appropriately

### Examples:
{examples}

### Java Code to Translate:
{query}

### C# Translation:"""
        else:
            source_lang, target_lang = "C#", "Java"
            template = """You are an expert code translator. Translate the given C# code to idiomatic Java code.

Rules:
- Return ONLY the translated Java code without explanations
- Maintain the same functionality and logic
- Use Java naming conventions (camelCase for methods)
- Use appropriate Java syntax and idioms
- Handle C#-specific constructs appropriately

### Examples:
{examples}

### C# Code to Translate:
{query}

### Java Translation:"""
        
        # Format examples
        examples_text = ""
        for i, (src, tgt) in enumerate(similar_examples):
            examples_text += f"\nExample {i+1}:\n{source_lang}:\n{src}\n\n{target_lang}:\n{tgt}\n"
        
        return template.format(examples=examples_text, query=query)
    
    def call_llm(self, 
                prompt: str, 
                max_tokens: int = 300, 
                temperature: float = 0.5) -> str:
        """
        Call the LLM for code translation.
        
        Args:
            prompt: The formatted prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Generated translation
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert code translator that produces accurate, idiomatic translations between programming languages."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise Exception(f"Failed to generate translation: {e}")
    
    def post_process_translation(self, translation: str, direction: str) -> str:
        """
        Post-process the translation to clean it up.
        
        Args:
            translation: Raw translation from LLM
            direction: Translation direction
            
        Returns:
            Cleaned translation
        """
        # Remove common artifacts
        translation = translation.strip()
        
        # Remove markdown code blocks if present
        if translation.startswith("```"):
            lines = translation.split('\n')
            if len(lines) > 2:
                translation = '\n'.join(lines[1:-1])
        
        # Remove language indicators
        for lang in ["java", "csharp", "c#", "cs"]:
            if translation.lower().startswith(lang):
                translation = translation[len(lang):].strip()
        
        # Remove common prompt artifacts
        artifacts = [
            "### Java Translation:",
            "### C# Translation:",
            "Here is the translation:",
            "Here's the translation:",
            "Translation:"
        ]
        
        for artifact in artifacts:
            if artifact in translation:
                translation = translation.split(artifact)[-1].strip()
        
        return translation
    
    def translate(self, 
                 code: str, 
                 direction: str, 
                 top_k: int = 3,
                 max_tokens: int = 300,
                 temperature: float = 0.5) -> str:
        """
        Translate code from one language to another.
        
        Args:
            code: Source code to translate
            direction: Translation direction ("java-to-cs" or "cs-to-java")
            top_k: Number of similar examples to use
            max_tokens: Maximum tokens in translation
            temperature: Sampling temperature
            
        Returns:
            Translated code
        """
        if not code.strip():
            raise ValueError("Input code cannot be empty")
        
        if direction not in ["java-to-cs", "cs-to-java"]:
            raise ValueError(f"Invalid direction: {direction}")
        
        try:
            # Search for similar examples
            similar_examples = self.search_similar_examples(code, direction, top_k)
            
            # Format prompt
            prompt = self.format_prompt(code, direction, similar_examples)
            
            # Generate translation
            raw_translation = self.call_llm(prompt, max_tokens, temperature)
            
            # Post-process
            final_translation = self.post_process_translation(raw_translation, direction)
            
            logger.info("Translation completed successfully")
            return final_translation
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise

# Example usage and testing
if __name__ == "__main__":
    try:
        # Initialize translator
        translator = CodeTranslator()
        
        # Test Java to C# translation
        java_code = '''public class Example {
    public void sayHello() {
        System.out.println("Hello, World!");
    }
}'''
        
        print("Java Code:")
        print(java_code)
        print("\nTranslated C# Code:")
        cs_translation = translator.translate(java_code, "java-to-cs")
        print(cs_translation)
        
        # Test C# to Java translation
        cs_code = '''public class Example 
{
    public void SayHello() 
    {
        Console.WriteLine("Hello, World!");
    }
}'''
        
        print("\n" + "="*50)
        print("C# Code:")
        print(cs_code)
        print("\nTranslated Java Code:")
        java_translation = translator.translate(cs_code, "cs-to-java")
        print(java_translation)
        
    except Exception as e:
        print(f"Error: {e}")