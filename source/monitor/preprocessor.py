import json
import random
import logging


# sample preprocess_handler (to be implemented by customer)
# This is a trivial example, where we simply generate random values
# But customers can read the data from inference_record and trasnform it into
# a flattened json structure
def preprocess_handler(inference_record):
    #event_data = inference_record.event_data
    
    input_enc_type = inference_record.endpoint_input.encoding
    input_data = inference_record.endpoint_input.data.rstrip("\n")
    output_data = inference_record.endpoint_output.data.rstrip("\n")
    
    logging.info("input_enc_type", input_enc_type)
    logging.info("input_data", input_data)
    logging.info("output_data", output_data)
    #eventmedatadata = inference_record.event_metadata
    #custom_attribute = json.loads(eventmedatadata.custom_attribute[0]) if eventmedatadata.custom_attribute is not None else None
    #is_test = eval_test_indicator(custom_attribute) if custom_attribute is not None else True
    
    if input_enc_type == "CSV":
        
        outputs = output_data+','+input_data
        
        logging.info({str(i).zfill(20) : d for i, d in enumerate(outputs.split(","))})
        
        return {str(i).zfill(20) : d for i, d in enumerate(outputs.split(","))}
    elif input_enc_type == "JSON":  
        outputs = {**{LABEL: output_data}, **json.loads(input_data)}
        write_to_file(str(outputs), "log")
        return {str(i).zfill(20) : outputs[d] for i, d in enumerate(outputs)}
    else:
        raise ValueError(f"encoding type {input_enc_type} is not supported") 