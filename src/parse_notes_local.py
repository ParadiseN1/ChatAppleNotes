import argparse
from macnotesapp import NotesApp
import ollama
from termcolor import colored
from transformers import LlamaTokenizerFast

def structure_notes(notes: list[dict], tokenizer: LlamaTokenizerFast) -> str:
    txt_notes = ""
    for note in reversed(notes):
        txt_notes += f"{note.modification_date}\n\n {note.body}\n"

    tokens = tokenizer.encode(txt_notes)[-6800:]
    txt_notes = tokenizer.decode(tokens)
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
            "hf-internal-testing/llama-tokenizer",
            legacy=False
        )
    
    prompt = """
    You are helpful assistant that helps analize my notes.
    Here is my provided notes:
    {notes}
    notes are provided in raw format, please make references or respond to me only in human readable format


    Here is my first message:
    {user_message}
    """
    
    history = []
    i = 0
    assistant_response = None
    notes_txt = structure_notes(notes, tokenizer)

    while True:
        if assistant_response:
            history.append({'role': 'assistant', 'content': assistant_response })
        user_message = input(colored('You: ', 'blue'))
        if i == 0:
            history.append({'role': 'user', 'content': prompt.format(notes=notes_txt, user_message=user_message)})
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
