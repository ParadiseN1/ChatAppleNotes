import argparse
from macnotesapp import NotesApp
import ollama
from termcolor import colored
from transformers import LlamaTokenizerFast

def structure_notes(notes: list[dict]) -> str:
    txt_notes = ""
    for note in reversed(notes):
        txt_notes += f"{note.modification_date}\n\n {note.body}\n"
    return txt_notes

def calc_tokens(history: list[dict], tokenizer: LlamaTokenizerFast) -> int:
    total_text = ''
    for i in history:
        total_text += i['content']

    
    total_tokens = len(tokenizer.encode(total_text))

    return total_tokens

def main():
    notesapp = NotesApp()
    notes = notesapp.notes()
    tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )
    
    prompt = """
    Here is my provided apple notes:
    {notes}

    Please help me as much as you can and do what i ask:
    {user_message}
    """
    
    history = []
    i = 0
    assistant_response = None
    while True:
        if assistant_response:
            history.append({'role': 'assistant', 'content': assistant_response })
        user_message = input(colored('You: ', 'blue'))
        if i == 0:
            history.append({'role': 'user', 'content': prompt.format(notes=structure_notes(notes), user_message=user_message)})
        else:
            history.append({'role': 'user', 'content': user_message })
        
        cur_tokens = calc_tokens(history=history, tokenizer=tokenizer)
        if cur_tokens > 7000:
            history = history[:2] + history[4:]
        
        stream = ollama.chat(
            model='llama3',
            messages=history,
            stream=True,
        )
        assistant_response = ""
        print(colored("Assistant:", 'red'), end='')
        for chunk in stream:
            assistant_response += chunk['message']['content']
            print(colored(chunk['message']['content'], 'green'), end='', flush=True)
        
        print('\n'+colored(f"(tokens: {cur_tokens})", 'cyan'))
        
        i += 1

def cli():
    # parser = argparse.ArgumentParser(description="CLI for interacting with Apple Notes and Ollama.")
    # parser.add_argument('user_message', type=str, help='Initial user message to the assistant')
    # args = parser.parse_args()
    # main(args.user_message)
    main()

if __name__ == "__main__":
    cli()

#TODO:
# 1. Integrate with llama-index to index all of the notes
# 2. Chat in RAG mode
# 3. Possible idea: insert as much similar Notes as possible in context to get high recall