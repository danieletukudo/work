import json
import re

def extract_qa_pairs(transcript_file):
    """Extract Q&A pairs from LiveKit transcript and format as requested"""
    
    with open(transcript_file, 'r') as f:
        data = json.load(f)
    
    items = data['session_history']['items']
    
    qa_pairs = []
    pair_count = 1
    
    i = 0
    while i < len(items):
        item = items[i]
        role = item['role']
        content = item['content'][0] if item['content'] else ""
        
        if role == 'assistant':
            # Check if this looks like a question
            if any(q in content.lower() for q in ['?', 'how', 'what', 'when', 'where', 'why', 'can you', 'tell me', 'share']):
                # Start collecting the question (may span multiple messages)
                question_parts = [content.strip()]
                
                # Look ahead for more assistant messages that might be part of the same question
                j = i + 1
                while j < len(items) and items[j]['role'] == 'assistant':
                    next_content = items[j]['content'][0] if items[j]['content'] else ""
                    question_parts.append(next_content.strip())
                    j += 1
                
                # Combine all question parts
                full_question = " ".join(question_parts)
                
                # Now collect all consecutive user responses
                user_responses = []
                while j < len(items) and items[j]['role'] == 'user':
                    user_content = items[j]['content'][0] if items[j]['content'] else ""
                    user_responses.append(user_content.strip())
                    j += 1
                
                # Combine all user responses for this question
                if user_responses:
                    combined_response = " ".join(user_responses)
                    
                    # Create a Q&A pair object
                    qa_pair = {
                        f"question{pair_count}": {
                            "text": full_question
                        },
                        f"answer{pair_count}": {
                            "text": combined_response
                        }
                    }
                    qa_pairs.append(qa_pair)
                    pair_count += 1
                
                # Update index to continue from where we left off
                i = j
            else:
                i += 1
        else:
            i += 1


    # save_qa_json(qa_pairs, "qa_pairs.json")
    
    return qa_pairs

def save_qa_json(qa_pairs, output_file):
    """Save Q&A pairs to JSON file in the requested format"""
    job_transcript = {
        "job_transcript": qa_pairs
    }
    
    with open(output_file, 'w') as f:
        json.dump(job_transcript, f, indent=2)

if __name__ == "__main__":

    # extract_qa_pairs()
    # Extract Q&A pairs
    # qa_pairs = extract_qa_pairs("fi.json")
    #
    # # Print the results
    # print("Q&A Pairs extracted:")
    # print("=" * 50)
    # for i, pair in enumerate(qa_pairs, 1):
    #     print(f"Pair {i}:")
    #     for key, value in pair.items():
    #         print(f"  {key}: {value['text']}")
    #     print()
    #
    # # Save to file
    save_qa_json(qa_pairs, "qa_pairs.json")
    # print(f"Q&A pairs saved to qa_pairs.json")
