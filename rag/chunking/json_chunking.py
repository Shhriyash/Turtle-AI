import json
from typing import List, Dict, Any, Union
from langchain_text_splitters import RecursiveJsonSplitter
from datetime import datetime


class ConversationChunker:
    """Handles chunking of conversation JSON data for RAG storage"""
    
    def __init__(self, max_chunk_size: int = 200, overlap_size: int = 80):
        """
        Initialize conversation chunker
        
        Args:
            max_chunk_size: Maximum tokens per chunk (~300 tokens)
            overlap_size: Overlap between chunks (~100 tokens) 
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
        # Initialize LangChain RecursiveJsonSplitter
        self.json_splitter = RecursiveJsonSplitter(
            max_chunk_size=max_chunk_size,
            min_chunk_size=50  # Minimum chunk size to avoid tiny fragments
        )
    
    def chunk_session_conversations(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a complete session's conversations
        
        Args:
            session_data: Session data in format:
                {
                    "session_id": {
                        "creation_time": "timestamp",
                        "conversations": [
                            {"user_query": "...", "turtle_response": "..."},
                            ...
                        ]
                    }
                }
        
        Returns:
            List of conversation chunks with metadata
        """
        all_chunks = []
        
        for session_id, session_info in session_data.items():
            conversations = session_info.get("conversations", [])
            creation_time = session_info.get("creation_time", datetime.now().isoformat())
            
            if not conversations:
                continue
            
            # Create conversational flow text for chunking
            conversation_flow = self._create_conversation_flow(conversations)
            
            # Chunk the conversation flow
            chunks = self._chunk_conversation_flow(conversation_flow, session_id, creation_time)
            all_chunks.extend(chunks)
        
        return all_chunks

    def chunk_turn_records(
        self,
        session_id: str,
        turn_records: List[Dict[str, Any]],
        creation_time: str,
    ) -> List[Dict[str, Any]]:
        """Chunk structured turn records without breaking turn boundaries."""
        chunks: List[Dict[str, Any]] = []
        if not turn_records:
            return chunks

        current_records: List[Dict[str, Any]] = []
        current_tokens = 0
        chunk_index = 0

        for turn_index, record in enumerate(turn_records):
            formatted = self._format_turn_record(record)
            if not formatted:
                continue

            estimated_tokens = max(1, len(formatted) // 4)
            if current_records and current_tokens + estimated_tokens > self.max_chunk_size:
                chunks.append(
                    self._build_turn_chunk(
                        session_id=session_id,
                        creation_time=creation_time,
                        chunk_index=chunk_index,
                        records=current_records,
                    )
                )
                chunk_index += 1
                current_records = []
                current_tokens = 0

            current_records.append(
                {
                    "turn_index": turn_index,
                    "formatted": formatted,
                    "kind": record.get("kind", "turn"),
                    "timestamp": record.get("timestamp"),
                }
            )
            current_tokens += estimated_tokens

        if current_records:
            chunks.append(
                self._build_turn_chunk(
                    session_id=session_id,
                    creation_time=creation_time,
                    chunk_index=chunk_index,
                    records=current_records,
                )
            )

        return chunks
    
    def _create_conversation_flow(self, conversations: List[Dict[str, str]]) -> str:
        """
        Convert conversation list to continuous conversational flow text
        
        Args:
            conversations: List of conversation dictionaries
            
        Returns:
            Continuous conversation text
        """
        flow_parts = []
        
        for i, conv in enumerate(conversations):
            user_query = conv.get("user_query", "")
            turtle_response = conv.get("turtle_response", "")
            
            # Create natural conversation flow
            conversation_part = f"User: {user_query}\nTurtle: {turtle_response}"
            flow_parts.append(conversation_part)
        
        # Join with double newlines to maintain conversation boundaries
        return "\n\n".join(flow_parts)
    
    def _chunk_conversation_flow(self, conversation_flow: str, session_id: str, creation_time: str) -> List[Dict[str, Any]]:
        """
        Chunk conversation flow text into overlapping segments
        
        Args:
            conversation_flow: Continuous conversation text
            session_id: Session identifier
            creation_time: Session creation timestamp
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        # Split conversation flow into approximate token-sized chunks
        # Estimate: ~4 characters per token
        chars_per_chunk = self.max_chunk_size * 4
        chars_overlap = self.overlap_size * 4
        
        text_length = len(conversation_flow)
        start = 0
        chunk_index = 0
        
        while start < text_length:
            # Calculate end position
            end = min(start + chars_per_chunk, text_length)
            
            # Find a good break point (end of sentence or conversation)
            if end < text_length:
                # Look for conversation boundaries first
                break_points = ['\n\n', '\n', '. ', '? ', '! ']
                for bp in break_points:
                    last_bp = conversation_flow.rfind(bp, start, end)
                    if last_bp > start:
                        end = last_bp + len(bp)
                        break
            
            # Extract chunk
            chunk_text = conversation_flow[start:end].strip()
            
            if chunk_text and len(chunk_text) > 20:  # Only add meaningful chunks
                chunk = {
                    "chunk_id": f"{session_id}_chunk_{chunk_index:03d}",
                    "session_id": session_id,
                    "creation_time": creation_time,
                    "chunk_index": chunk_index,
                    "content": chunk_text,
                    "start_pos": start,
                    "end_pos": end,
                    "char_count": len(chunk_text),
                    "estimated_tokens": len(chunk_text) // 4
                }
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap, but ensure progress
            next_start = end - chars_overlap
            if next_start <= start:  # Prevent infinite loop
                next_start = start + chars_per_chunk // 2  # Move forward by half chunk
            
            start = min(next_start, text_length)
        
        return chunks

    def _build_turn_chunk(
        self,
        *,
        session_id: str,
        creation_time: str,
        chunk_index: int,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        content = "\n\n".join(record["formatted"] for record in records).strip()
        tags = sorted({str(record.get("kind", "turn")) for record in records})
        first_turn = records[0]["turn_index"]
        last_turn = records[-1]["turn_index"]
        return {
            "chunk_id": f"{session_id}_turn_chunk_{chunk_index:03d}",
            "session_id": session_id,
            "creation_time": creation_time,
            "chunk_index": chunk_index,
            "kind": "turn_chunk",
            "turn_range": [first_turn, last_turn],
            "timestamp_start": records[0].get("timestamp"),
            "timestamp_end": records[-1].get("timestamp"),
            "tags": tags,
            "content": content,
            "char_count": len(content),
            "estimated_tokens": len(content) // 4,
        }

    def _format_turn_record(self, record: Dict[str, Any]) -> str:
        label_map = {
            "user": "User",
            "assistant": "Turtle",
            "tool_call": "Assistant tool call",
            "tool_return": "Tool result",
            "retry": "Validation feedback",
        }
        label = label_map.get(str(record.get("kind", "turn")), "Turn")
        tool_name = str(record.get("tool_name", "")).strip()
        content = str(record.get("content", "")).strip()
        if not content:
            return ""
        if tool_name:
            return f"{label} ({tool_name}): {content}"
        return f"{label}: {content}"
    
    def chunk_json_data(self, json_data: Union[Dict, str]) -> List[Dict[str, Any]]:
        """
        Chunk JSON data using LangChain's RecursiveJsonSplitter
        
        Args:
            json_data: JSON data as dict or string
            
        Returns:
            List of JSON chunks
        """
        # Convert string to dict if needed
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        
        try:
            # Use LangChain's RecursiveJsonSplitter
            json_chunks = self.json_splitter.split_json(json_data=json_data)
            
            # Add metadata to chunks
            processed_chunks = []
            for i, chunk in enumerate(json_chunks):
                processed_chunk = {
                    "chunk_id": f"json_chunk_{i:03d}",
                    "chunk_index": i,
                    "content": json.dumps(chunk, ensure_ascii=False),
                    "content_dict": chunk,
                    "estimated_tokens": len(str(chunk)) // 4
                }
                processed_chunks.append(processed_chunk)
            
            return processed_chunks
            
        except Exception as e:
            return []
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about chunks
        
        Args:
            chunks: List of chunks
            
        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {"total_chunks": 0}
        
        char_counts = [chunk.get("char_count", len(chunk.get("content", ""))) for chunk in chunks]
        token_counts = [chunk.get("estimated_tokens", 0) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chars_per_chunk": sum(char_counts) / len(char_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "min_chars": min(char_counts) if char_counts else 0,
            "max_chars": max(char_counts) if char_counts else 0,
            "total_chars": sum(char_counts),
            "total_estimated_tokens": sum(token_counts)
        }


# Global chunker instance
conversation_chunker = None

def get_chunker() -> ConversationChunker:
    """Get global conversation chunker instance"""
    global conversation_chunker
    if conversation_chunker is None:
        conversation_chunker = ConversationChunker()
    return conversation_chunker

def chunk_conversations(session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convenience function to chunk conversations"""
    chunker = get_chunker()
    return chunker.chunk_session_conversations(session_data)

def chunk_json(json_data: Union[Dict, str]) -> List[Dict[str, Any]]:
    """Convenience function to chunk JSON data"""
    chunker = get_chunker()
    return chunker.chunk_json_data(json_data)


