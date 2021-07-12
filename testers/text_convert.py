import os
TEXT_FILE_PATH = "intervju/"
TEXT_FILE_PATH_WRITE = "intervju/converted/" # Do not change

def convert_characters(text):
    text_converted = text
    text_converted = text_converted.replace("\\\\u00e5", "å")
    text_converted = text_converted.replace("\\\\u00e4", "ä")
    text_converted = text_converted.replace("\\\\u00f6", "ö")
    text_converted = text_converted.replace("&#39;", "'")
    text_converted = text_converted.replace('"[{\\\"who\\\": \\\"bot\\\", \\\"message\\\": \\\"', "")
    text_converted = text_converted.replace('\\\"}, {\\\"who\\\": \\\"bot\\\", \\\"message\\\": \\\"', "\n")
    text_converted = text_converted.replace('\\\"}, {\\\"who\\\": \\\"user\\\", \\\"message\\\": \\\"', "\n")
    text_converted = text_converted.replace('\\\"}]"', "")
    text_converted = text_converted.replace('\\\", \\\"message_en\\\": \\\"', "\n")

    text_array = text_converted.split('\n')
    text_array = text_array[1::2]
    text_converted = "\n".join(text_array)
    return text_converted


if __name__ == '__main__':
    # Create directory for converted files
    if not os.path.exists(TEXT_FILE_PATH_WRITE):
        os.makedirs(TEXT_FILE_PATH_WRITE)

    for text_file_name in [f for f in os.listdir(TEXT_FILE_PATH) if not f.startswith('.') and not f.startswith("converted")]:
        print("Processing ",text_file_name, "... ")
        text_file_read = open(TEXT_FILE_PATH + text_file_name,'r')
        text = text_file_read.read()
        text_converted = convert_characters(text)
        #print(text_converted)
        text_file_write = open(TEXT_FILE_PATH_WRITE + text_file_name + "_converted",'w')
        text_file_write.write(text_converted)
        print("Done!\n")
