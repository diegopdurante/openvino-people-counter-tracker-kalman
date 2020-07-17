#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.net_plugin = None
        self.exec_network = None
        self.input_blob = None
        self.output_blob = None
        self.output_blobs = []
        
        self.infer_request_handle = None

    def load_model(self, model, device="CPU", extensions = None, debug = False):
        
        # Set model definition and file and replace extension for bin file
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        if debug:
            print(model_xml, model_bin)
        
        # 1
        self.plugin = IECore()
        
        # TODO Complete this part
        # Pendiente
        if extensions and "CPU" in device:
            self.plugin.add_extension(extensions, device)
            if debug:
                print("Extension added")
#         if len(extensions) > 0:
#             for extension in extensions:
#                 self.plugin.add_extension(extension, device)
#                 print("Added CPU extension: ", extension)

        # Create the network plugin
        self.net_plugin = IENetwork(model = model_xml, weights = model_bin)

        ### Check for supported layers
        supported_layers = self.plugin.query_network(self.net_plugin, device)
        not_supported_layers = [l for l in self.net_plugin.layers.keys() if l not in supported_layers]
        
#         # TODO: Pending: Está bien esto?

        if(len(not_supported_layers) > 0):
            print("Error: Unsupported layers")
            print(not_supported_layers)
            return False
        else:
            if debug:
                print("OK! All layers are supported")
        if debug:
            print("IR successfully loaded into Inference Engine.")
        
        ### TODO: Add any necessary extensions ###
        # TODO: Número de requests
        self.exec_network = self.plugin.load_network(self.net_plugin, device)
        
        # Calculate input and output blobs (to calculate shapes)
        self.input_blob = next(iter(self.net_plugin.inputs))
        self.output_blob = next(iter(self.net_plugin.outputs))
        
        # For models like yolo (multiple outputs)
        for it in self.net_plugin.outputs:
#             print(it)
            self.output_blobs.append(it)

        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return True
        #self.plugin = IEPlugin(device=device)
    def get_input_shape(self):
        # Return the shape of the input layer
        return self.net_plugin.inputs[self.input_blob].shape

    def exec_net(self, image, req_id = 0):
        # Start an asynchronous request
        # TODO: Pending: Ver request
        self.infer_request_handle = self.exec_network.start_async(request_id=req_id, inputs={self.input_blob: image})
        return self.infer_request_handle

    def wait(self, req_id = 0, timeout = -1):
        # Wait for the request to be complete.
        # timeout =  0 - Immediately returns the inference status. It does not block or interrupt execution.
        # timeout = -1 - Waits until inference result becomes available (default value)
        status = self.exec_network.requests[req_id].wait(timeout)
        return status

    def get_output(self, req_id = 0, debug = False):
        # TODO: Modificar el número de request (cur_request id)
#         for i in self.net_plugin.outputs:
#             print (i)
        if debug == True:
            pass # TODO: completar
#         print(self.output_blob)
#         print(self.net_plugin.outputs)
        return self.exec_network.requests[req_id].outputs[self.output_blob]

    def get_outputs(self, debug = False):
        
        retval = []
        
        for each in self.output_blobs:
            
            if debug == True:
                print(each)
                
            retval.append(self.exec_network.requests[0].outputs[each])
            
        return retval
    
    def clean(self):
        
        del self.net_plugin
        del self.plugin
        del self.network