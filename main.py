import os
#from langchain_community.llms import Ollama
from langchain_ollama.llms import OllamaLLM as Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
#from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM as Ollama
#from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

def setup_rag_system(documents_folder):
    # Set up the RAG system with all PDFs from a folder using local Ollama    
    print(f"Loading all PDFs from '{documents_folder}' folder...")
    loader = PyPDFDirectoryLoader(documents_folder)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from all PDFs in the folder")
    
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks")
    
    print("Creating embeddings with local Ollama...")
    print("(This may take a few minutes for the first run)")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )
    
    print("Building vector database...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("Setting up language model...")
    llm = Ollama(
        model="llama3.2",  # also can be used "mistral", "phi3", "gemma2", etc.
        temperature=0.3
    )
    
    # Create custom prompt
    prompt_template = """Use the following context to answer the question. 
If you don't know the answer based on the context, say "I cannot find this information in the document."

Context: {context}

Question: {question}

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create RAG chain
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = RunnableParallel(
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "source_documents": retriever
        }
    ).assign(
        result=lambda x: (PROMPT | llm | StrOutputParser()).invoke({
            "context": x["context"],
            "question": x["question"]
        })
    )
    
    print("\n✅ RAG System ready!\n")
    return qa_chain

def ask_question(qa_chain, question):
    """
    Ask a question to the RAG system
    """
    #print(f"\n{'='*60}")
    #print(f"Question: {question}")
    #print(f"{'='*60}")
    print("Thinking...")
    
    result = qa_chain.invoke(question)
    
    print(f"Answer: {result['result']}")
    print(f"\n📄 Sources: {len(result['source_documents'])} document chunks used")
    print(f"{'='*60}\n")
    return result

def main():
    print("🤖 RAG System Interactive Mode (Local Ollama)")
    """
    Main function to run the RAG system
    """
    # Configuration
    DOCUMENTS_FOLDER = "documents"  # Folder containing your PDF files
    
    # Set up the RAG system
    qa_chain = setup_rag_system(DOCUMENTS_FOLDER)
    
    # Interactive Q&A loop
    print("🤖 RAG System Interactive Mode (Local Ollama)")
    print("Type your questions about the PDFs")
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
            
        if not question:
            print("Please enter a question.\n")
            continue
            
        res = ask_question(qa_chain, question)
        #metadata = []
        #for _ in res['source_documents']:
        #    metadata.append(('page: ' + str(_.metadata['page']), _.metadata[DOCUMENTS_FOLDER]))

        #print(f"Asistent:", res['result'], "\n", metadata)
            

if __name__ == "__main__":
    main()