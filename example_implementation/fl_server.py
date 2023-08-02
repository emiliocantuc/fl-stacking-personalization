"""
Record bandwith usage

Page rank for importances??

Poisonning islands
"""


from concurrent import futures
import grpc,threading,random,datetime,argparse,sys,pickle,os,shutil
sys.path.insert(0, '..')
import utils
import networkx as nx

import example_implementation_pb2,example_implementation_pb2_grpc

class FLServer(example_implementation_pb2_grpc.FLServerServicer):
    """
    Example 'aggregator' server.
    """
    # Reported island importances. island reporting -> {other_island: importance,...}
    island_importances={}

    computed_island_importances={}

    plain_island_importances={}

    # 
    models_info={}

    # Directory that holds submitted models
    model_dir='models/'


    # To avoid registering two islands under the same name
    admission_lock=threading.Lock()

    def __init__(self) -> None:
        super().__init__()
        print('Im alive')
        self.clear(None,None)


    def request_admission(self,request,context):
        """
        Admits an island to the federation. 
        As an example we assume all islands that request entry are admitted.
        """

        try:

            # Aquire lock - don't want two islands with the same username
            FLServer.admission_lock.acquire()

            # We check if the island's name is already registered
            if request.name in FLServer.island_importances:

                # Release lock
                FLServer.admission_lock.release()

                print(f'Request admission: {request.name} already exists.')
                return example_implementation_pb2.Status(success=False,details='Try again with a different username')
            
            # Add island
            FLServer.island_importances[request.name]={}

            # And release lock
            FLServer.admission_lock.release()

            print(f'Request admission: admitted {request.name}')

            return example_implementation_pb2.Status(success=True)
        
        except Exception as e:
            print(f'Error admitting: {e}')
            return example_implementation_pb2.Status(success=False,details='Server error.')
        

    def submit_model(self,request,context):
        """
        Receives a model submission, decides wheter to accept it,
        and stores the model along with its information.
        """
        # Only registered islands can submit a model
        if request.from_island not in FLServer.island_importances:
            return example_implementation_pb2.Status(success=False,details='Register before submitting a model')

        try:
            # TODO how to verify identity
            FLServer.models_info[request.from_island]=request.info

            # Save model
            utils.savePickeObj(request.model.model,FLServer.model_dir,request.from_island)

            print(f'Models submitted by {request.from_island}')

            return example_implementation_pb2.Status(success=True)

        except Exception as e:
            print(f'Error processing a model submission: {e}')
            return example_implementation_pb2.Status(success=False,details='Server error.')
        
    
    def submit_importances(self,request,context):
        """
        Receives an importances submission, decides wheter to accept it,
        and stores it.
        """
        # Only registered islands can submit importances
        if request.from_island not in FLServer.island_importances:
            return example_implementation_pb2.Status(success=False,details='Register before submitting importances')

        try:
            # TODO how to verify identity

            # Island -> importance assined by request.from_island
            imp_dict={island:imp for island,imp in zip(request.importances.islands,request.importances.importances)}

            # Check that importances add to 1
            if abs(1-sum(imp_dict.values()))>1e-3:
                print(f'Importances do not add to 1 ({sum(imp_dict.values())})')
                return example_implementation_pb2.Status(success=False,details='Importances must add to 1')
            
            # Ignore reported value on self and islands not registered
            imp_dict={i:v for i,v in imp_dict.items() if i!=request.from_island and i in FLServer.island_importances}

            # Insert into graph
            if request.from_island not in FLServer.island_importances:

                FLServer.island_importances[request.from_island]=imp_dict
            else:

                # Overwrite existing but maintain old
                for other_islad,imp in imp_dict.items():
                    FLServer.island_importances[request.from_island][other_islad]=imp

                # Renormalize
                norm=sum(FLServer.island_importances[request.from_island].values())
                for other_island,imp in FLServer.island_importances[request.from_island].items():
                    FLServer.island_importances[request.from_island][other_island]=imp/norm

                if abs(1-sum(FLServer.island_importances[request.from_island].values()))>=1e-3:
                    print('Importances do not sum to 1')


            print(f'Importances submitted by {request.from_island}: {imp_dict}')

            # TODO trigger computing island importances
            FLServer.compute_importances()

            return example_implementation_pb2.Status(success=True)

        except Exception as e:
            print(f'Error processing an importances submission: {e}')
            return example_implementation_pb2.Status(success=False,details='Server error.')
        
    def get_importances(self,request,context):

        try:
            islands=list(FLServer.computed_island_importances.keys())
            importances=[FLServer.computed_island_importances[i] for i in islands]
            plain=[FLServer.plain_island_importances[i] for i in islands]
            return example_implementation_pb2.Importances(islands=islands,importances=importances,plain_importances=plain)
        
        except Exception as e:
            print(f'Error processing importances fetch: {e}')
            pass


    def get_model_info(self,request,context):

        try:
            # Defaults to all
            models=request.models if len(request.models)>0 else list(FLServer.models_info.keys())

            info=[FLServer.models_info[model] for model in models if model in FLServer.models_info]

            return example_implementation_pb2.ModelInfoReply(
                status=example_implementation_pb2.Status(success=True),
                info=info
            )

        except Exception as e:
            print(f'Error serving models info: {e}')
            return example_implementation_pb2.ModelInfoReply(
                status=example_implementation_pb2.Status(success=False,details='Server Error.')
            )


    def get_models(self,request,context):

        try:
            # Defaults to all except own TODO
            models=request.models if len(request.models)>0 else list(FLServer.models_info.keys())

            models=[m for m in models if m in FLServer.models_info]

            info=[FLServer.models_info[m] for m in models]

            model_bytes=[
                example_implementation_pb2.Model(model=pickle.dumps(utils.loadPickeObj(FLServer.model_dir,m)))
                for m in models
            ]

            return example_implementation_pb2.ModelsFetchReply(
                status=example_implementation_pb2.Status(success=True),
                models_info=info,
                models=model_bytes
            )

        except Exception as e:
            print(f'Error serving models info: {e}')
            return example_implementation_pb2.ModelsFetchReply(
                status=example_implementation_pb2.Status(success=False,details='Server Error.')
            )
        
    def clear(self,request,context):
        # Delete models, importantes, etc
        shutil.rmtree(FLServer.model_dir,ignore_errors=True)

        FLServer.island_importances={}
        FLServer.computed_island_importances={}
        FLServer.plain_island_importances={}
        FLServer.models_info={}

        print('Server Cleared')
        return example_implementation_pb2.Status(success=True)


    def compute_importances():
        # Construct graph using plain island importances
        # Island -> sum of incoming weights to it
        plain={i:0 for i in FLServer.island_importances}

        graph=[]
        for island, importances in FLServer.island_importances.items():
            for other_island,weight in importances.items():
                graph.append([island,other_island,{'weight':weight}])
                plain[other_island]+=weight

        G = nx.DiGraph(graph)
        FLServer.computed_island_importances = nx.pagerank(G)
        print('Computed island importances w/page rank: ',FLServer.computed_island_importances)

        plain_norm=sum(plain.values())
        plain={i:imp/plain_norm for i,imp in plain.items()}
        FLServer.plain_island_importances = plain
        print('Computed plain importances: ',plain)

    



def launch_server(port):
    """
    Launches the server on the specified port.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),options = [
        ('grpc.max_send_message_length', int(1e9)),
        ('grpc.max_receive_message_length', int(1e9))
    ])
    
    example_implementation_pb2_grpc.add_FLServerServicer_to_server(FLServer(), server)
    #server.add_secure_port()
    server.add_insecure_port("[::]:" + port)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":

    # Parse the port to launch the server
    parser=argparse.ArgumentParser(
        prog='Example implementation of an aggregator server',
        description='',
    )
    parser.add_argument('-p','--port',default='50052',help='Port to launch the server onto')
    args=vars(parser.parse_args())

    print("Launching server ...")
    launch_server(args['port'])