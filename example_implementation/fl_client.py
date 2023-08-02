import grpc,sys,argparse,os,pickle
import example_implementation_pb2,example_implementation_pb2_grpc

class Model:
    def __init__(self,s) -> None:
        self.s=s


CHANNEL_OPTIONS=[
        ('grpc.max_send_message_length', int(1e9)),
        ('grpc.max_receive_message_length', int(1e9))
    ]

class Client():

    def __init__(self,name,server_url):
        self.name=name
        self.server_url=server_url

    def request_admission(self):
        with grpc.insecure_channel(self.server_url) as channel:
            stub=example_implementation_pb2_grpc.FLServerStub(channel)
            stub.request_admission(example_implementation_pb2.AddmissionRequest(name=self.name))

    def submit_model(self,model,trained_on_n,model_type):
        # TODO check model is fitted

        # Serialize the model using pickle
        serialized_model=pickle.dumps(model)

        # Get model size in bytes
        model_bytes=len(serialized_model)

        # Open a connection
        with grpc.insecure_channel(self.server_url,options=CHANNEL_OPTIONS) as channel:

            # Get stub
            stub=example_implementation_pb2_grpc.FLServerStub(channel)

            # Call submit model
            stub.submit_model(example_implementation_pb2.ModelSubmission(
                from_island=self.name,
                info=example_implementation_pb2.ModelInfo(from_island=self.name,trained_on_n=trained_on_n,bytes=model_bytes,model_type=model_type),
                model=example_implementation_pb2.Model(model=pickle.dumps(model))
            ))

    def submit_importances(self,importances):
        # Un pack dictionary into lists

        if len(importances)>0:
            islands, importances_ = zip(*importances.items())
            
            # Open a connection
            with grpc.insecure_channel(self.server_url,options=CHANNEL_OPTIONS) as channel:

                # Get stub
                stub=example_implementation_pb2_grpc.FLServerStub(channel)

                # Call submit importances
                stub.submit_importances(example_implementation_pb2.ImportancesSubmission(
                    from_island=self.name,
                    importances=example_implementation_pb2.Importances(islands=islands,importances=importances_),
                    
                ))

    def get_importances(self):

        
        # Open a connection
        with grpc.insecure_channel(self.server_url,options=CHANNEL_OPTIONS) as channel:

            # Get stub
            stub=example_implementation_pb2_grpc.FLServerStub(channel)

            # Call get importances
            imps=stub.get_importances(example_implementation_pb2.Empty())

            page_rank_imps=dict(zip(imps.islands,imps.importances))
            plain_imps=dict(zip(imps.islands,imps.plain_importances))
            
            return {'page_rank':page_rank_imps,'plain':plain_imps}

    def get_model_info(self):
        out={}

        # Open a connection
        with grpc.insecure_channel(self.server_url,options=CHANNEL_OPTIONS) as channel:

            # Get stub
            stub=example_implementation_pb2_grpc.FLServerStub(channel)

            # Call get model info 
            reply=stub.get_model_info(example_implementation_pb2.ModelsFetchRequest())

            for model_info_reply in reply.info:
                from_island=model_info_reply.from_island
                trained_on_n=model_info_reply.trained_on_n
                bytes=model_info_reply.bytes
                model_type=model_info_reply.model_type

                out[from_island]={'trained_on_n':trained_on_n,'bytes':bytes,'model_type':model_type}

            return out

    def get_models(self,models):

        out={}

        # Open a conection
        with grpc.insecure_channel(self.server_url,options=CHANNEL_OPTIONS) as channel:

            # Get stub
            stub=example_implementation_pb2_grpc.FLServerStub(channel)

            # For every model in the response
            response=stub.get_models(example_implementation_pb2.ModelsFetchRequest(models=models))

            for m_info,m in zip(response.models_info,response.models):

                # Unserialize it using pickle
                model=pickle.loads(pickle.loads(m.model))
                out[m_info.from_island]=model
                
            return out
        
    def clear_server(self):

        out={}

        # Open a conection
        with grpc.insecure_channel(self.server_url,options=CHANNEL_OPTIONS) as channel:

            # Get stub
            stub=example_implementation_pb2_grpc.FLServerStub(channel)

            # For every model in the response
            response=stub.clear(example_implementation_pb2.Empty())

            if response.success:
                print('Server cleared')
        
    

if __name__=='__main__':
    """
    import fl_client

    m=MyScikitLearnModel().fit(X,y)

    fl=fl_client.Client(URL,id='A',token=SSL_TOKEN)

    # If access is granted to the federation
    if fl.request_admission():
    
        fl.submit_model(model=m,meta_data=None)

        meta_data=fl.fetch_meta_data()

        fl.fetch_models()

        importances=train_meta_model()

        fl.submit_importances(importances)
        
    """

    parser=argparse.ArgumentParser(
        prog='Example of an island interface with the federation',
    )
    parser.add_argument('-u','--url',default='localhost:50052',help='The federations IP adress')
    args=vars(parser.parse_args())

    clients={i:Client(i,args['url']) for i in 'abcd'}

    for c in clients.values():
        c.request_admission()

    clients['a'].submit_importances({'b':0.5,'c':0.5})
    clients['a'].submit_importances({'d':1.0})
    clients['a'].clear_server()



    