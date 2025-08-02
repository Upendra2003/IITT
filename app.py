import streamlit as st
import sys
import os
from pathlib import Path

# Add the current directory to Python path to import model
sys.path.append(str(Path(__file__).parent))

try:
    from model import CodeTranslator
except ImportError:
    st.error("Could not import model.py. Please ensure model.py is in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Code Translator",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 0.5rem;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .code-container {
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
        padding: 1rem;
        background-color: #f8f9fa;
    }
    .translation-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem 0.5rem 0 0;
        margin: -1rem -1rem 1rem -1rem;
        font-weight: 600;
    }
    .error-container {
        background-color: #fee;
        border: 1px solid #fcc;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-container {
        background-color: #efe;
        border: 1px solid #cfc;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e7f3ff;
        color: #000000;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'translator' not in st.session_state:
    st.session_state.translator = None
if 'translation_history' not in st.session_state:
    st.session_state.translation_history = []

def initialize_translator():
    """Initialize the translator with error handling"""
    try:
        with st.spinner("🔄 Loading models and indices... This may take a moment."):
            translator = CodeTranslator()
            return translator
    except Exception as e:
        st.error(f"❌ Failed to initialize translator: {str(e)}")
        st.markdown("""
        <div class="info-box">
        <strong>Troubleshooting:</strong><br>
        • Ensure all required files are present (retrieval_mapping.pkl, cs_vectors.npy, java_vectors.npy)<br>
        • Check your internet connection for model downloads<br>
        • Verify that all dependencies are installed
        </div>
        """, unsafe_allow_html=True)
        return None

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="color: #667eea; margin-bottom: 0.5rem;">🔄 Code Translator</h1>
        <p style="color: #666; font-size: 1.1rem;">Translate between Java and C# with AI-powered assistance</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for settings and info
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Translation direction
        direction = st.selectbox(
            "Translation Direction",
            options=["java-to-cs", "cs-to-java"],
            format_func=lambda x: "Java → C#" if x == "java-to-cs" else "C# → Java",
            help="Select the direction of code translation"
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            max_tokens = st.slider(
                "Max Tokens", 
                min_value=100, 
                max_value=1000, 
                value=300,
                help="Maximum number of tokens in the translation"
            )
            
            temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5,
                help="Controls randomness in translation (lower = more deterministic)"
            )
            
            top_k = st.slider(
                "Similar Examples", 
                min_value=1, 
                max_value=5, 
                value=3,
                help="Number of similar examples to use for context"
            )

        # Model info
        st.markdown("### 📊 Model Info")
        st.info("""
        **Embedding Model:** microsoft/codebert-base  
        **Translation Model:** llama-3.3-70b-versatile  
        **Search:** FAISS vector similarity
        """)

        # History
        if st.session_state.translation_history:
            st.markdown("### 📝 Recent Translations")
            for i, (src, tgt, dir_used) in enumerate(reversed(st.session_state.translation_history[-3:])):
                with st.expander(f"Translation {len(st.session_state.translation_history) - i}"):
                    st.markdown(f"**Direction:** {dir_used}")
                    st.code(src[:100] + "..." if len(src) > 100 else src)

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        source_lang = "Java" if direction == "java-to-cs" else "C#"
        st.markdown(f"### 📝 {source_lang} Code")
        
        # Example codes for quick testing
        example_java = '''public class HelloWorld {
    public void sayHello() {
        System.out.println("Hello, World!");
    }
    
    public int add(int a, int b) {
        return a + b;
    }
}'''

        example_cs = '''public class HelloWorld 
{
    public void SayHello() 
    {
        Console.WriteLine("Hello, World!");
    }
    
    public int Add(int a, int b) 
    {
        return a + b;
    }
}'''

        # Quick example buttons
        st.markdown("**Quick Examples:**")
        col_ex1, col_ex2 = st.columns(2)
        
        with col_ex1:
            if st.button("📄 Use Example", key="example1"):
                st.session_state.input_code = example_java if source_lang == "Java" else example_cs
        
        with col_ex2:
            if st.button("🗑️ Clear", key="clear"):
                st.session_state.input_code = ""

        # Code input area
        input_code = st.text_area(
            f"Enter your {source_lang} code:",
            height=300,
            placeholder=f"Paste your {source_lang} code here...",
            key="input_code"
        )

    with col2:
        target_lang = "C#" if direction == "java-to-cs" else "Java"
        st.markdown(f"### 🎯 {target_lang} Translation")
        
        # Initialize translator if not already done
        if st.session_state.translator is None:
            st.session_state.translator = initialize_translator()

        # Translation button
        if st.button("🚀 Translate Code", disabled=not input_code.strip()):
            if st.session_state.translator is None:
                st.error("❌ Translator not initialized. Please refresh the page and try again.")
            else:
                try:
                    with st.spinner("🔄 Translating..."):
                        # Show progress
                        progress_bar = st.progress(0)
                        progress_bar.progress(25)
                        
                        translated_code = st.session_state.translator.translate(
                            input_code, 
                            direction, 
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_k=top_k
                        )
                        progress_bar.progress(100)
                        
                        # Store in history
                        st.session_state.translation_history.append((input_code, translated_code, direction))
                        
                        # Display result
                        st.markdown('<div class="translation-header">✅ Translation Complete</div>', unsafe_allow_html=True)
                        st.code(translated_code, language='csharp' if target_lang == 'C#' else 'java')
                        
                        # Copy button
                        st.markdown("**Copy translated code:**")
                        st.code(translated_code)
                        
                        progress_bar.empty()
                        
                except Exception as e:
                    st.markdown(f'<div class="error-container">❌ <strong>Translation Error:</strong><br>{str(e)}</div>', unsafe_allow_html=True)
        
        # Display placeholder when no translation
        if not input_code.strip():
            st.markdown('<div class="code-container">', unsafe_allow_html=True)
            st.markdown("🔄 **Translation will appear here**")
            st.markdown("Enter code in the left panel and click 'Translate Code' to get started.")
            st.markdown('</div>', unsafe_allow_html=True)

    # Footer with tips
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <strong>💡 Tips for better translations:</strong><br>
    • Provide complete, well-formatted code snippets<br>
    • Include necessary imports and class declarations<br>
    • Use meaningful variable and method names<br>
    • Test the translated code in your target environment
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()