import argparse
import datetime
import os
import sys
import whisper


def transcribe_file(filename: str, model: str = 'large') -> str:
    """
    Transcribe the audio file at the given path and return the transcription as a string.

    Args:
        filename (str): The path to the audio file.
        model (str, optional): The model to use for transcription. Defaults to 'large'.

    Returns:
        str: The transcription of the audio file.
    """
    result = whisper.transcribe(filename, model=model)
    return result.get('text')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, nargs='+', help='Audio files to transcribe')
    args = parser.parse_args()

    print('Loading large model...')
    whisper.load_models.load_model('large')

    for filename in args.input_file:
        if not os.path.exists(filename):
            print(f'The input file {filename} does not exist.', file=sys.stderr)
            continue
        output = os.path.splitext(filename)[0] + '.txt'
        print(f'Transcribing audio file {filename} to {output}... ', end='', flush=True)
        start_time = datetime.datetime.now()
        text = transcribe_file(filename, 'large')
        end_time = datetime.datetime.now()
        print(f'done in {end_time-start_time}.')
        text = text.strip()
        if text:
            with open(output, 'w') as outfile:
                outfile.write(text)
        else:
            print(f'No transcription for {filename}.', file=sys.stderr)


if __name__ == '__main__':
    main()
