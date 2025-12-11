# RAG Chatbot with Separate Table and Text Processing + Reinforcement Learning from Chat History
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from typing import List, Tuple, Dict
import json
import re
import pandas as pd
import tabula.io as tabula
import os
import pickle
from datetime import datetime
from collections import Counter
import requests



class RAGChatbot:
    def __init__(self, pdf_path: str, hf_token: str):
        self.pdf_path = pdf_path
        self.hf_token = hf_token
        self.chunks = []
        self.chunk_metadata = []
        self.index = None
        self.embeddings_model = None
        
        # ✅ NEW: API configuration
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        self.headers = {"Authorization": f"Bearer {hf_token}"}
        self.model_name = "meta-llama/Llama-3.3-70B-Instruct:sambanova"
        
        self.chat_history = []
        self.output_dir = "./"
        self.table_csv_path = None
        self.text_chunks_path = None
        self.history_file = os.path.join(self.output_dir, "chat_history.pkl")
        
        self.chat_embeddings = []
        self.chat_index = None
        self.chat_embedding_file = os.path.join(self.output_dir, "chat_embeddings.pkl")
        
        self.query_patterns = Counter()
        self.feedback_scores = {}
        self.stats_file = os.path.join(self.output_dir, "learning_stats.pkl")
        
        self.conversation_context = {
            'current_employee': None,
            'last_mentioned_entities': []
        }
    
        os.makedirs(self.output_dir, exist_ok=True)
    
        self._load_chat_history()
        self._load_learning_stats()
    
        self._setup()
        
        self._build_chat_history_index()

    def _load_chat_history(self):
        """Load chat history from file if exists"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'rb') as f:
                    self.chat_history = pickle.load(f)
                print(f"Loaded {len(self.chat_history)} previous conversations")
            except Exception as e:
                print(f"Could not load chat history: {e}")
                self.chat_history = []
        else:
            self.chat_history = []

    def _save_chat_history(self):
        """Save chat history to file"""
        try:
            with open(self.history_file, 'wb') as f:
                pickle.dump(self.chat_history, f)
        except Exception as e:
            print(f"Could not save chat history: {e}")

    def _load_learning_stats(self):
        """Load learning statistics"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'rb') as f:
                    data = pickle.load(f)
                    self.query_patterns = data.get('query_patterns', Counter())
                    self.feedback_scores = data.get('feedback_scores', {})
                print(f"Loaded learning statistics: {len(self.query_patterns)} patterns tracked")
            except Exception as e:
                print(f"Could not load learning stats: {e}")
                self.query_patterns = Counter()
                self.feedback_scores = {}
        else:
            self.query_patterns = Counter()
            self.feedback_scores = {}

    def _save_learning_stats(self):
        """Save learning statistics"""
        try:
            with open(self.stats_file, 'wb') as f:
                pickle.dump({
                    'query_patterns': self.query_patterns,
                    'feedback_scores': self.feedback_scores
                }, f)
        except Exception as e:
            print(f"Could not save learning stats: {e}")

    def _build_chat_history_index(self):
        """Build FAISS index from chat history for semantic search"""
        if len(self.chat_history) == 0:
            print("No chat history to index")
            return

        print(f"Building semantic index for {len(self.chat_history)} past conversations...")

        # Create embeddings for all past Q&A pairs
        chat_texts = []
        for entry in self.chat_history:
            # Combine question and answer for better context
            combined_text = f"Q: {entry['question']}\nA: {entry['answer']}"
            chat_texts.append(combined_text)

        # Generate embeddings
        self.chat_embeddings = self.embeddings_model.encode(chat_texts, show_progress_bar=True)

        # Build FAISS index
        dimension = self.chat_embeddings.shape[1]
        self.chat_index = faiss.IndexFlatL2(dimension)
        self.chat_index.add(np.array(self.chat_embeddings).astype('float32'))

        # Save embeddings
        try:
            with open(self.chat_embedding_file, 'wb') as f:
                pickle.dump(self.chat_embeddings, f)
        except Exception as e:
            print(f"Could not save chat embeddings: {e}")

        print(f"Chat history index built successfully")

    def _search_chat_history(self, query: str, k: int = 5) -> List[Dict]:
        """Search through past conversations semantically"""
        if self.chat_index is None or len(self.chat_history) == 0:
            return []

        # Encode query
        query_embedding = self.embeddings_model.encode([query])

        # Search
        distances, indices = self.chat_index.search(
            np.array(query_embedding).astype('float32'),
            min(k, len(self.chat_history))
        )

        # Return relevant past conversations
        relevant_chats = []
        for idx, distance in zip(indices[0], distances[0]):
            if distance < 1.5:  # Similarity threshold
                relevant_chats.append({
                    'chat': self.chat_history[idx],
                    'similarity_score': float(distance)
                })

        return relevant_chats

    def _extract_entities_from_query(self, query: str) -> Dict:
        """Extract names and entities from query"""
        query_lower = query.lower()

        # Check for pronouns that need context
        has_pronoun = bool(re.search(r'\b(his|her|their|he|she|they|him|them)\b', query_lower))

        # Try to extract names (capitalize words that might be names)
        potential_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)

        return {
            'has_pronoun': has_pronoun,
            'names': potential_names
        }

    def _update_conversation_context(self, question: str, answer: str):
        """Update context tracking based on conversation"""
        # Extract names from question
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)

        # Extract names from answer
        answer_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer)

        # Update current employee if employee was mentioned
        if 'employee' in answer.lower() or 'working' in answer.lower():
            all_names = names + answer_names
            if all_names:
                self.conversation_context['current_employee'] = all_names[0]
                # Keep last 5 mentioned entities
                self.conversation_context['last_mentioned_entities'] = (
                    all_names[:5] if len(all_names) <= 5
                    else self.conversation_context['last_mentioned_entities'][-4:] + [all_names[0]]
                )

    def _resolve_pronouns(self, query: str) -> str:
        """Replace pronouns with actual entity names from context"""
        entities = self._extract_entities_from_query(query)

        if entities['has_pronoun'] and self.conversation_context['current_employee']:
            current_name = self.conversation_context['current_employee']

            # Replace pronouns with the current employee name
            query = re.sub(r'\bhis\b', f"{current_name}'s", query, flags=re.IGNORECASE)
            query = re.sub(r'\bher\b', f"{current_name}'s", query, flags=re.IGNORECASE)
            query = re.sub(r'\bhe\b', current_name, query, flags=re.IGNORECASE)
            query = re.sub(r'\bshe\b', current_name, query, flags=re.IGNORECASE)

        return query


    def _extract_query_pattern(self, query: str) -> str:
        """Extract pattern from query for learning"""
        query_lower = query.lower()

        # Detect common patterns
        patterns = []

        if re.search(r'\bhow many\b', query_lower):
            patterns.append('count_query')
        if re.search(r'\bwho\b', query_lower):
            patterns.append('who_query')
        if re.search(r'\bwhat\b', query_lower):
            patterns.append('what_query')
        if re.search(r'\bwhen\b', query_lower):
            patterns.append('when_query')
        if re.search(r'\bwhere\b', query_lower):
            patterns.append('where_query')
        if re.search(r'\blist\b|\ball\b', query_lower):
            patterns.append('list_query')
        if re.search(r'\bcalculate\b|\bsum\b|\btotal\b|\baverage\b', query_lower):
            patterns.append('calculation_query')
        if re.search(r'\bemployee\b|\bstaff\b|\bworker\b', query_lower):
            patterns.append('employee_query')
        if re.search(r'\bpolicy\b|\brule\b|\bguideline\b', query_lower):
            patterns.append('policy_query')

        return '|'.join(patterns) if patterns else 'general_query'

    def _load_pdf_text(self) -> str:
        """Load text from PDF"""
        text = ""
        with open(self.pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def _extract_and_merge_tables(self) -> str:
        """Extract all tables from PDF and merge into single CSV"""
        try:
            print("Extracting tables from PDF...")

            # Extract all tables
            dfs = tabula.read_pdf(self.pdf_path, pages="all", multiple_tables=True)

            if not dfs or len(dfs) == 0:
                print("No tables found in PDF")
                return None

            print(f"Found {len(dfs)} tables")

            # The first table has headers
            merged_df = dfs[0]

            # Append rest of the tables
            for i in range(1, len(dfs)):
                # Set the column names to match the first table
                dfs[i].columns = merged_df.columns
                # Append rows
                merged_df = pd.concat([merged_df, dfs[i]], ignore_index=True)

            # Save merged table
            csv_path = os.path.join(self.output_dir, "merged_employee_tables.csv")
            merged_df.to_csv(csv_path, index=False)

            print(f"Merged {len(dfs)} tables into {csv_path}")
            print(f"Total rows: {len(merged_df)}")
            print(f"Columns: {list(merged_df.columns)}")

            return csv_path

        except Exception as e:
            print(f"Table extraction failed: {e}")
            return None

    def _save_table_chunks(self, table_chunks: List[Dict]) -> str:
        """Save table chunks (full table + row chunks) to a text file"""
        save_path = os.path.join(self.output_dir, "table_chunks.txt")

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"Total Table Chunks: {len(table_chunks)}\n")
            f.write("=" * 80 + "\n\n")

            for i, chunk in enumerate(table_chunks):
                f.write(f"CHUNK {i + 1} [Type: {chunk['type']}]\n")
                f.write("-" * 80 + "\n")
                f.write(chunk['content'])
                f.write("\n\n" + "=" * 80 + "\n\n")

        print(f"Saved {len(table_chunks)} table chunks to {save_path}")
        return save_path

    def _detect_table_regions_in_text(self, text: str) -> List[Tuple[int, int]]:
        """Detect start and end positions of table regions in text"""
        lines = text.split('\n')
        table_regions = []
        start_idx = None

        for i, line in enumerate(lines):
            is_table_line = (
                    '@' in line or
                    re.search(r'\b(A|B|AB|O)[+-]\b', line) or
                    re.search(r'\s{3,}', line) or
                    re.search(r'Employee Name|Email|Position|Table|Blood Group', line, re.IGNORECASE)
            )

            if is_table_line:
                if start_idx is None:
                    start_idx = i
            else:
                if start_idx is not None:
                    # End of table region
                    if i - start_idx > 3:  # Only consider tables with 3+ lines
                        table_regions.append((start_idx, i))
                    start_idx = None

        # Handle last table if exists
        if start_idx is not None and len(lines) - start_idx > 3:
            table_regions.append((start_idx, len(lines)))

        return table_regions

    def _remove_table_text(self, text: str) -> str:
        """Remove table content from text"""
        lines = text.split('\n')
        table_regions = self._detect_table_regions_in_text(text)

        # Create set of line indices to remove
        lines_to_remove = set()
        for start, end in table_regions:
            for i in range(start, end):
                lines_to_remove.add(i)

        # Keep only non-table lines
        clean_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]

        return '\n'.join(clean_lines)

    def _chunk_text_content(self, text: str) -> List[Dict]:
        """Chunk text content (Q&A pairs and other text)"""
        chunks = []

        # Remove table text
        clean_text = self._remove_table_text(text)

        # Split by ###Question###
        qa_pairs = clean_text.split('###Question###')

        for i, qa in enumerate(qa_pairs):
            if not qa.strip():
                continue

            if '###Answer###' in qa:
                chunk_text = '###Question###' + qa
                if len(chunk_text) > 50:
                    chunks.append({
                        'content': chunk_text,
                        'type': 'qa',
                        'source': 'text_content',
                        'chunk_id': f'qa_{i}'
                    })

        # Also create chunks from sections (for non-Q&A content)
        sections = re.split(r'\n\n+', clean_text)
        for i, section in enumerate(sections):
            section = section.strip()
            if len(section) > 200 and '###Question###' not in section:
                chunks.append({
                    'content': section,
                    'type': 'text',
                    'source': 'text_content',
                    'chunk_id': f'text_{i}'
                })

        return chunks

    def _save_text_chunks(self, chunks: List[Dict]) -> str:
        """Save text chunks to file"""
        text_path = os.path.join(self.output_dir, "text_chunks.txt")

        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"Total Text Chunks: {len(chunks)}\n")
            f.write("=" * 80 + "\n\n")

            for i, chunk in enumerate(chunks):
                f.write(f"CHUNK {i + 1} [Type: {chunk['type']}]\n")
                f.write("-" * 80 + "\n")
                f.write(chunk['content'])
                f.write("\n\n" + "=" * 80 + "\n\n")

        print(f"Saved {len(chunks)} text chunks to {text_path}")
        return text_path

    def _load_csv_as_text(self, csv_path: str) -> str:
        """Load CSV and convert to readable text format"""
        try:
            df = pd.read_csv(csv_path)
            text = f"[EMPLOYEE TABLE DATA]\n"
            text += f"Total Employees: {len(df)}\n\n"
            text += df.to_string(index=False)
            return text
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return ""

    def _create_table_chunks(self, csv_path: str) -> List[Dict]:
        """Create chunks from CSV table"""
        chunks = []

        try:
            df = pd.read_csv(csv_path)

            # Create one chunk with full table overview
            full_table_text = f"[COMPLETE EMPLOYEE TABLE]\n"
            full_table_text += f"Total Employees: {len(df)}\n"
            full_table_text += f"Columns: {', '.join(df.columns)}\n\n"
            full_table_text += df.to_string(index=False)

            chunks.append({
                'content': full_table_text,
                'type': 'table_full',
                'source': 'employee_table.csv',
                'chunk_id': 'table_full'
            })

            # Create chunks for each row (employee)
            for idx, row in df.iterrows():
                row_text = f"[EMPLOYEE RECORD {idx + 1}]\n"
                for col in df.columns:
                    row_text += f"{col}: {row[col]}\n"

                chunks.append({
                    'content': row_text,
                    'type': 'table_row',
                    'source': 'employee_table.csv',
                    'chunk_id': f'employee_{idx}'
                })

            print(f"Created {len(chunks)} chunks from table ({len(df)} employee records + 1 full table)")

        except Exception as e:
            print(f"Error creating table chunks: {e}")

        return chunks

    def _save_manifest(self, all_chunks: List[Dict]):
        """Save manifest of all chunks"""
        manifest = {
            'total_chunks': len(all_chunks),
            'chunks_by_type': {
                'qa': sum(1 for c in all_chunks if c['type'] == 'qa'),
                'text': sum(1 for c in all_chunks if c['type'] == 'text'),
                'table_full': sum(1 for c in all_chunks if c['type'] == 'table_full'),
                'table_row': sum(1 for c in all_chunks if c['type'] == 'table_row')
            },
            'files_created': {
                'table_csv': self.table_csv_path,
                'text_chunks': self.text_chunks_path
            },
            'chunk_details': [
                {
                    'chunk_id': c['chunk_id'],
                    'type': c['type'],
                    'source': c['source'],
                    'length': len(c['content'])
                }
                for c in all_chunks
            ]
        }

        manifest_path = os.path.join(self.output_dir, 'chunk_manifest.json')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"Saved manifest to {manifest_path}")
        return manifest_path

    def _resolve_pronouns_for_session(self, query: str, conversation_context: Dict) -> str:
        """Resolve pronouns using session-specific context"""
        entities = self._extract_entities_from_query(query)

        if entities['has_pronoun'] and conversation_context.get('current_employee'):
            current_name = conversation_context['current_employee']

            query = re.sub(r'\bhis\b', f"{current_name}'s", query, flags=re.IGNORECASE)
            query = re.sub(r'\bher\b', f"{current_name}'s", query, flags=re.IGNORECASE)
            query = re.sub(r'\bhe\b', current_name, query, flags=re.IGNORECASE)
            query = re.sub(r'\bshe\b', current_name, query, flags=re.IGNORECASE)

        return query

    def _search_session_history(self, query: str, session_history: List[Dict], k: int = 5) -> List[Dict]:
        """Search through session-specific history"""
        if not session_history:
            return []

        chat_texts = [f"Q: {entry['question']}\nA: {entry['answer']}" for entry in session_history]

        if not chat_texts:
            return []

        chat_embeddings = self.embeddings_model.encode(chat_texts)

        dimension = chat_embeddings.shape[1]
        temp_index = faiss.IndexFlatL2(dimension)
        temp_index.add(np.array(chat_embeddings).astype('float32'))

        query_embedding = self.embeddings_model.encode([query])
        distances, indices = temp_index.search(
            np.array(query_embedding).astype('float32'),
            min(k, len(session_history))
        )

        relevant_chats = []
        for idx, distance in zip(indices[0], distances[0]):
            if distance < 1.5:
                relevant_chats.append({
                    'chat': session_history[idx],
                    'similarity_score': float(distance)
                })

        return relevant_chats

    def _build_prompt_for_session(self, query: str, retrieved_data: List[Tuple[str, Dict]],
                                  relevant_past_chats: List[Dict], session_history: List[Dict],
                                  conversation_context: Dict) -> str:
        """Build prompt using session-specific data"""

        employee_records = []
        full_table = []
        qa_context = []
        text_context = []

        for content, metadata in retrieved_data:
            if metadata['type'] == 'table_row':
                employee_records.append(content)
            elif metadata['type'] == 'table_full':
                full_table.append(content)
            elif metadata['type'] == 'qa':
                qa_context.append(content)
            elif metadata['type'] == 'text':
                text_context.append(content)

        context_text = ""
        if full_table:
            context_text += "COMPLETE EMPLOYEE TABLE:\n" + "\n".join(full_table) + "\n\n"
        if employee_records:
            context_text += "RELEVANT EMPLOYEE RECORDS:\n" + "\n\n".join(employee_records[:15]) + "\n\n"
        if qa_context:
            context_text += "COMPANY POLICIES & Q&A:\n" + "\n\n".join(qa_context) + "\n\n"
        if text_context:
            context_text += "ADDITIONAL INFORMATION:\n" + "\n\n".join(text_context)

        context_memory = ""
        if conversation_context.get('current_employee'):
            context_memory = f"\nCURRENT CONVERSATION CONTEXT:\n"
            context_memory += f"Currently discussing: {conversation_context['current_employee']}\n"
            if conversation_context.get('last_mentioned_entities'):
                context_memory += f"Recently mentioned: {', '.join(conversation_context['last_mentioned_entities'])}\n"
            context_memory += "\n"

        past_context = ""
        if relevant_past_chats:
            past_context += "RELEVANT PAST CONVERSATIONS (for context):\n"
            for i, chat_info in enumerate(relevant_past_chats[:3], 1):
                chat = chat_info['chat']
                past_context += f"\n[Past Q&A {i}]:\n"
                past_context += f"Previous Question: {chat['question']}\n"
                past_context += f"Previous Answer: {chat['answer']}\n"
            past_context += "\n"

        history_text = ""
        for entry in session_history[-10:]:
            history_text += f"User: {entry['question']}\nAssistant: {entry['answer']}\n\n"

        prompt = f"""<s>[INST] You are a helpful HR assistant for Acme AI Ltd. Use the provided context to answer questions accurately.

    IMPORTANT INSTRUCTIONS:
    - You have access to COMPLETE EMPLOYEE TABLE and individual employee records
    - For employee-related queries, use the employee data provided
    - If you find any name from user input, always look into the EMPLOYEE TABLE first
    - PAY ATTENTION to pronouns (his, her, he, she) - they refer to people mentioned in THIS USER's recent conversation
    - When user asks about "his email" or "her position", look at the conversation context to understand who they're referring to
    - Be careful not to give all employee information - only answer what was asked
    - For counting or calculations, use the table data
    - For policy questions, use the Q&A knowledge base
    - Provide specific, accurate answers based on the context
    - If information is not in the context, say "I don't have this information"
    - Round up any fractional numbers in calculations

    Context:
    {context_text}

    {context_memory}

    {past_context}

    Recent conversation:
    {history_text}

    User Question: {query}

    Answer based on the context above. Be specific and accurate.[/INST]"""

        return prompt

    def _update_conversation_context_for_session(self, question: str, answer: str, conversation_context: Dict):
        """Update session-specific conversation context"""
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
        answer_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer)

        if 'employee' in answer.lower() or 'working' in answer.lower():
            all_names = names + answer_names
            if all_names:
                conversation_context['current_employee'] = all_names[0]
                conversation_context['last_mentioned_entities'] = (
                    all_names[:5] if len(all_names) <= 5
                    else conversation_context.get('last_mentioned_entities', [])[-4:] + [all_names[0]]
                )

    def _setup(self):
        print("\n" + "=" * 80)
        print("STEP 1: Loading PDF")
        print("=" * 80)
    
        text = self._load_pdf_text()
        print(f"Loaded PDF with {len(text)} characters")
    
        print("\n" + "=" * 80)
        print("STEP 2: Extracting and Merging Tables")
        print("=" * 80)
    
        self.table_csv_path = self._extract_and_merge_tables()
    
        print("\n" + "=" * 80)
        print("STEP 3: Chunking Text Content (Removing Tables)")
        print("=" * 80)
    
        text_chunks = self._chunk_text_content(text)
        self.text_chunks_path = self._save_text_chunks(text_chunks)
    
        print("\n" + "=" * 80)
        print("STEP 4: Creating Final Chunks")
        print("=" * 80)
    
        all_chunks = []
        all_chunks.extend(text_chunks)
    
        if self.table_csv_path:
            table_chunks = self._create_table_chunks(self.table_csv_path)
            all_chunks.extend(table_chunks)
            self._save_table_chunks(table_chunks)
    
        self.chunks = [c['content'] for c in all_chunks]
        self.chunk_metadata = all_chunks
    
        print(f"\nTotal chunks created: {len(self.chunks)}")
        print(f"  - Q&A chunks: {sum(1 for c in all_chunks if c['type'] == 'qa')}")
        print(f"  - Text chunks: {sum(1 for c in all_chunks if c['type'] == 'text')}")
        print(f"  - Table full: {sum(1 for c in all_chunks if c['type'] == 'table_full')}")
        print(f"  - Employee records: {sum(1 for c in all_chunks if c['type'] == 'table_row')}")
    
        self._save_manifest(all_chunks)
    
        print("\n" + "=" * 80)
        print("STEP 5: Creating Embeddings")
        print("=" * 80)
    
        print("Loading embedding model...")
        self.embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
        print("Creating embeddings for all chunks...")
        embeddings = self.embeddings_model.encode(self.chunks, show_progress_bar=True)
    
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
    
        print("\n" + "=" * 80)
        print("STEP 6: Initializing LLM API")
        print("=" * 80)
        
        # ✅ API already configured in __init__
        print(f"API URL: {self.api_url}")
        print(f"Model: {self.model_name}")
        print("LLM API ready!")
    
        print("\n" + "=" * 80)
        print("SETUP COMPLETE!")
        print("=" * 80)

    def _retrieve(self, query: str, k: int = 10) -> List[Tuple[str, Dict]]:
        """Retrieve relevant chunks with metadata"""
        query_embedding = self.embeddings_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)

        results = []
        for idx in indices[0]:
            results.append((self.chunks[idx], self.chunk_metadata[idx]))

        return results

    def _build_prompt(self, query: str, retrieved_data: List[Tuple[str, Dict]], relevant_past_chats: List[Dict]) -> str:
        """Build prompt with retrieved context and learned information from past chats"""

        # Separate different types of context
        employee_records = []
        full_table = []
        qa_context = []
        text_context = []

        for content, metadata in retrieved_data:
            if metadata['type'] == 'table_row':
                employee_records.append(content)
            elif metadata['type'] == 'table_full':
                full_table.append(content)
            elif metadata['type'] == 'qa':
                qa_context.append(content)
            elif metadata['type'] == 'text':
                text_context.append(content)

        # Build context sections
        context_text = ""

        if full_table:
            context_text += "COMPLETE EMPLOYEE TABLE:\n" + "\n".join(full_table) + "\n\n"

        if employee_records:
            context_text += "RELEVANT EMPLOYEE RECORDS:\n" + "\n\n".join(employee_records[:15]) + "\n\n"

        if qa_context:
            context_text += "COMPANY POLICIES & Q&A:\n" + "\n\n".join(qa_context) + "\n\n"

        if text_context:
            context_text += "ADDITIONAL INFORMATION:\n" + "\n\n".join(text_context)

        # ADD THIS NEW SECTION:
        context_memory = ""
        if self.conversation_context['current_employee']:
            context_memory = f"\nCURRENT CONVERSATION CONTEXT:\n"
            context_memory += f"Currently discussing: {self.conversation_context['current_employee']}\n"
            if self.conversation_context['last_mentioned_entities']:
                context_memory += f"Recently mentioned: {', '.join(self.conversation_context['last_mentioned_entities'])}\n"
            context_memory += "\n"

        # Build relevant past conversations (learning from history)
        past_context = ""
        if relevant_past_chats:
            past_context += "RELEVANT PAST CONVERSATIONS (for context):\n"
            for i, chat_info in enumerate(relevant_past_chats[:3], 1):
                chat = chat_info['chat']
                past_context += f"\n[Past Q&A {i}]:\n"
                past_context += f"Previous Question: {chat['question']}\n"
                past_context += f"Previous Answer: {chat['answer']}\n"
            past_context += "\n"

        # CHANGE THIS LINE from [-3:] to [-10:]:
        history_text = ""
        for entry in self.chat_history:  # Changed from -3 to -10
            history_text += f"User: {entry['question']}\nAssistant: {entry['answer']}\n\n"

        prompt = f"""<s>[INST] You are a helpful HR assistant for Acme AI Ltd. Use the provided context to answer questions accurately.

    IMPORTANT INSTRUCTIONS:
    - You have access to COMPLETE EMPLOYEE TABLE and individual employee records
    - For employee-related queries, use the employee data provided
    - If you find any name from user input, always look into the EMPLOYEE TABLE first. If you still can't find, then you can go for chunked text.
    - PAY ATTENTION to pronouns (his, her, he, she) - they refer to people mentioned in recent conversation
    - When user asks about "his email" or "her position", look at the conversation context to understand who they're referring to
    - While your answer is related to an employee, be careful of not giving all the information of the employee. Just give the information user asked.
    - For counting or calculations, use the table data
    - For policy questions, use the Q&A knowledge base
    - LEARN from relevant past conversations - if similar questions were asked before, maintain consistency
    - Use patterns from past interactions to improve answer quality
    - Provide specific, accurate answers based on the context
    - If you need to count employees or perform calculations, do it carefully from the data
    - If information is not in the context, just say "I don't have this information in the provided documents"
    - While performing any type of mathematical calculation, always round up any fractional number.

    Context:
    {context_text}

    {context_memory}

    {past_context}

    Recent conversation:
    {history_text}

    User Question: {query}

    Answer based on the context above. Be specific and accurate. But don't always start with "based on the context"[/INST]"""

        return prompt

    def ask(self, question: str) -> str:
        """Ask a question to the chatbot with learning from past conversations"""
        if question.lower() in ["reset data", "reset"]:
            self.chat_history = []
            self.chat_embeddings = []
            self.chat_index = None
            self.conversation_context = {'current_employee': None, 'last_mentioned_entities': []}
            self._save_chat_history()
            return "Chat history has been reset."
    
        # Resolve pronouns before processing
        resolved_question = self._resolve_pronouns(question)
        
        # Extract query pattern for learning
        pattern = self._extract_query_pattern(resolved_question)
        self.query_patterns[pattern] += 1
    
        # Search through past conversations for similar questions
        relevant_past_chats = self._search_chat_history(resolved_question, k=5)
    
        # Retrieve relevant chunks
        retrieved_data = self._retrieve(resolved_question, k=20)
    
        # Build prompt
        prompt = self._build_prompt(resolved_question, retrieved_data, relevant_past_chats)
    
        # ✅ NEW: Call Hugging Face Router API
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 512,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            # Extract answer from response
            answer = result["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            answer = "I apologize, but I'm having trouble generating a response right now. Please try again."
    
        # Update conversation context
        self._update_conversation_context(question, answer)
    
        # Store in history
        chat_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'pattern': pattern,
            'used_past_context': len(relevant_past_chats) > 0
        }
        
        self.chat_history.append(chat_entry)
    
        # Update chat history index
        new_text = f"Q: {question}\nA: {answer}"
        new_embedding = self.embeddings_model.encode([new_text])
        
        if self.chat_index is None:
            dimension = new_embedding.shape[1]
            self.chat_index = faiss.IndexFlatL2(dimension)
            self.chat_embeddings = new_embedding
        else:
            self.chat_embeddings = np.vstack([self.chat_embeddings, new_embedding])
        
        self.chat_index.add(np.array(new_embedding).astype('float32'))
    
        # Save to disk
        self._save_chat_history()
        self._save_learning_stats()
    
        return answer

    def provide_feedback(self, question: str, rating: int):
        """Allow user to rate responses for reinforcement learning (1-5 scale)"""
        if 1 <= rating <= 5:
            # Find the most recent occurrence of this question
            for i in range(len(self.chat_history) - 1, -1, -1):
                if self.chat_history[i]['question'] == question:
                    chat_id = f"{i}_{self.chat_history[i]['timestamp']}"
                    self.feedback_scores[chat_id] = rating
                    self._save_learning_stats()
                    print(f"Feedback recorded: {rating}/5")
                    return
            print("Question not found in recent history")
        else:
            print("Rating must be between 1 and 5")

    def get_learning_insights(self) -> Dict:
        """Get insights about what the chatbot has learned"""
        total_conversations = len(self.chat_history)
        conversations_with_past_context = sum(
            1 for c in self.chat_history if c.get('used_past_context', False)
        )

        avg_feedback = 0
        if self.feedback_scores:
            avg_feedback = sum(self.feedback_scores.values()) / len(self.feedback_scores)

        return {
            'total_conversations': total_conversations,
            'conversations_using_past_context': conversations_with_past_context,
            'query_patterns': dict(self.query_patterns.most_common(10)),
            'total_feedback_entries': len(self.feedback_scores),
            'average_feedback_score': round(avg_feedback, 2)
        }

    def get_history(self) -> List[Dict]:
        """Get chat history"""
        return self.chat_history

    def display_stats(self):
        """Display system statistics"""
        qa_chunks = sum(1 for c in self.chunk_metadata if c['type'] == 'qa')
        text_chunks = sum(1 for c in self.chunk_metadata if c['type'] == 'text')
        table_full = sum(1 for c in self.chunk_metadata if c['type'] == 'table_full')
        table_rows = sum(1 for c in self.chunk_metadata if c['type'] == 'table_row')

        insights = self.get_learning_insights()

        print(f"\n{'=' * 80}")
        print("CHATBOT STATISTICS")
        print(f"{'=' * 80}")
        print(f"Total chunks: {len(self.chunks)}")
        print(f"  - Q&A chunks: {qa_chunks}")
        print(f"  - Text chunks: {text_chunks}")
        print(f"  - Full table: {table_full}")
        print(f"  - Employee records: {table_rows}")
        print(f"\nLEARNING STATISTICS:")
        print(f"  - Total conversations: {insights['total_conversations']}")
        print(f"  - Conversations using past context: {insights['conversations_using_past_context']}")
        print(f"  - Total feedback entries: {insights['total_feedback_entries']}")
        print(f"  - Average feedback score: {insights['average_feedback_score']}/5")
        print(f"\nTop query patterns:")
        for pattern, count in list(insights['query_patterns'].items())[:5]:
            print(f"  - {pattern}: {count}")
        print(f"\nOutput directory: {self.output_dir}/")
        print(f"Table CSV: {os.path.basename(self.table_csv_path) if self.table_csv_path else 'None'}")
        print(f"Text chunks: {os.path.basename(self.text_chunks_path)}")
        print(f"History file: {os.path.basename(self.history_file)}")
        print(f"Learning stats: {os.path.basename(self.stats_file)}")
        print(f"{'=' * 80}\n")


# Main execution
if __name__ == "__main__":
    # Configuration
    PDF_PATH = "./data/policies.pdf"
    HF_TOKEN = os.getenv("HF_TOKEN")

    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not set")

    # Initialize chatbot
    print("\nInitializing RAG Chatbot with Learning Capabilities...")
    bot = RAGChatbot(PDF_PATH, HF_TOKEN)

    # Display statistics
    bot.display_stats()

    # Chat loop
    print("Chatbot ready! Type 'exit' to quit, 'stats' for learning insights, or 'feedback' to rate last answer.\n")
    last_question = None

    while True:
        user_input = input("You: ")

        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break

        if user_input.lower() == 'stats':
            insights = bot.get_learning_insights()
            print("\nLearning Insights:")
            print(json.dumps(insights, indent=2))
            continue

        if user_input.lower() == 'feedback':
            if last_question:
                try:
                    rating = int(input("Rate the last answer (1-5): "))
                    bot.provide_feedback(last_question, rating)
                except ValueError:
                    print("Invalid rating")
            else:
                print("No previous question to rate")
            continue

        if not user_input.strip():
            continue

        try:
            last_question = user_input
            answer = bot.ask(user_input)
            print(f"\nBot: {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")