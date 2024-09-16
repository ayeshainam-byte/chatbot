from PyPDF2 import PdfReader
from langchain_community.llms import CTransformers
import dspy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Defining the model
class LLaMALLM:
    def __init__(self):
        self.llm = CTransformers(
            model="TheBloke/Llama-2-7B-Chat-GGML",
            model_type="llama",
            max_new_tokens=150,
            temperature=0.5,
        )
 
    def generate(self, prompt):
        return self.llm(prompt).strip()

llm = LLaMALLM()

# Define the Signature
class GenerateAnswer(dspy.Signature):
    """Answer questions with short answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="accurate and often between 1 and 5 words")

# Define the Module
class SimpleChatbotModule(dspy.Module):
    def __init__(self, pdf_path, num_passages=3):
        super().__init__()
        self.pdf_path = pdf_path
        self.num_passages = num_passages
        self.llm = llm 
        self.docs = self.load_pdf()

    def load_pdf(self):
        reader = PdfReader(self.pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text.split("\n")

    def simple_vectorizer(self, text):
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([text] + self.docs)
        return vectors.toarray()[0], vectors.toarray()[1:]

    def retrieve(self, query):
        """Retrieves the most relevant passages based on the query."""
        query_vector, doc_vectors = self.simple_vectorizer(query)
        scores = np.dot(doc_vectors, query_vector)
        top_indices = np.argsort(scores)[-self.num_passages:]
        return [self.docs[i] for i in top_indices]

    def forward(self, question):
        passages = self.retrieve(question)
        context_text = " ".join(passages)
        prompt = f"Answer the question using the provided context.\n\nContext: {context_text}\nQuestion: {question}\nAnswer:"
        answer = self.llm.generate(prompt)
        return dspy.Prediction(context=context_text, answer=answer)

def validate_response(example, pred):
    return dspy.evaluate.answer_exact_match(example, pred)

pdf_path = 'python.pdf'
trainset = [
    dspy.Example(question="How is Python used in automation?", answer="Python is used in automation through scripting and automation tools."),
    dspy.Example(question="What is python?", answer="Python is a computer programming language often used to build websites and software, automate tasks, and analyze data."),
]
chatbot = SimpleChatbotModule(pdf_path, num_passages=3)

def optimize_module(module, trainset):
    pass

optimize_module(chatbot, trainset)

question = "what are features of python?"
response = chatbot(question)
print(response.answer)