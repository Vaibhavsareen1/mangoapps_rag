import os
import requests
import gradio as gr
from typing import List, Tuple
from dotenv import load_dotenv

with gr.Blocks(fill_height=True) as application:
    """Main gradio block to store the entire RAG front-end application"""

    # Load enviornment variables from .env file
    load_dotenv()

    BACKEND_URL_BASE = "http://127.0.0.1:3000"

    # Split the interface into two parts
    # Part 1 to upload documents
    # Part 2 to chat with the LLM model
    with gr.Row(equal_height=True) as interface_split_row:
        with gr.Column(scale=1) as document_upload_column:
            upload_documents_box = gr.Files(scale=3)
            upload_document_button = gr.Button(value='Upload', scale=1)
    
        with gr.Column(scale=3) as chat_interface_column:
            chat_interface = gr.Chatbot(render_markdown=True,
                                        label='Conversation Area',
                                        show_copy_button=True,
                                        type='messages',
                                        scale=3)
            
            with gr.Row(equal_height=True) as input_row:
                user_query_box = gr.Textbox(interactive=True, label='User Query')
                generate_button = gr.Button(value='Generate')

    @upload_document_button.click(inputs=[upload_documents_box],
                                  outputs=[upload_documents_box])
    def upload_files(files: List[str]) -> List[str]:
        """
        Function to take user selected files and upload it to the fastapi-server.
        Files with extension pdf will only be uploaded onto the server.

        :Parameters:
        files: List of file names to be uploaded

        :Returns:
        An empty list to be rendered or the same list depending upon the completion
        of the process.
        """
        # Store required extensions
        extensions_dict = {'pdf': 'application/pdf'}
        # End point where the files are to be loaded
        upsert_url = '/'.join([BACKEND_URL_BASE, 'upsert'])
        # Instantiate a list to store invalid files
        invalid_files = []

        # Check if there are files to be uploaded else raise an error message
        if len(files) > 0:
            # Loop through the files and upload it
            for file in files:
                file_base_name = os.path.basename(file) 
                _, file_extension = file_base_name.split('.')
                # If the file extension is valid then send the post request
                if file_extension.lower() in extensions_dict:
                    files_payload = {'upload_file': (file_base_name, open(file, 'rb'), extensions_dict[file_extension])}
                    # Send the request
                    response = requests.post(upsert_url, files=files_payload)
                    print(f'{file_base_name}', response.status_code)
                else:
                    invalid_files.append(file)

            if len(invalid_files) > 0:
                gr.Info(message='Some files have not been uploaded. Please check the file extension.')
            else:
                gr.Info(message='All files have been uploaded successfully.')
            
            return invalid_files
        else:
            gr.Info(message="Please add files before clicking on the 'Upload' Button.")
            return []

    @generate_button.click(inputs=[user_query_box, chat_interface],
                           outputs=[user_query_box, chat_interface])
    def generate_ai_response(user_message: str, chat_history: List[gr.ChatMessage]) -> Tuple[str, List[gr.ChatMessage]]:
        """
        Function which gets triggered when user queries something from the documents

        :Parameters:
        user_message: Message by the user to be queried to extract information
        chat_history: History of previous chat messages

        :Returns:
        Tuple containing empty user query and previous chat history 
        """

        chat_history.append(gr.ChatMessage(role='user', content=user_message))
        json_payload = {'user_query': user_message}
        # End point where the files are to be loaded
        chat_url = '/'.join([BACKEND_URL_BASE, 'chat'])
        try:
            with requests.post(chat_url, json=json_payload, stream=True) as response:
                response.raise_for_status()

                # Initialize assistant's message in the chat
                assistant_message = gr.ChatMessage(role='assistant', content="")
                chat_history.append(assistant_message)

                # Stream the response content
                for chunk in response.iter_content(decode_unicode=True):
                    if chunk:
                        # Update the assistant's message content incrementally
                        assistant_message.content += chunk
                        yield '', chat_history
     
        except Exception as error:
            gr.Info(message='Retry Sending information')

        return '', chat_history
    

    if __name__ == '__main__':
        application.launch(server_name='127.0.0.1', server_port=8002)