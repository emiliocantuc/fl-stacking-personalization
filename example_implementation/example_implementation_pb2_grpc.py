# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import example_implementation_pb2 as example__implementation__pb2


class FLServerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.request_admission = channel.unary_unary(
                '/example_implementation.FLServer/request_admission',
                request_serializer=example__implementation__pb2.AddmissionRequest.SerializeToString,
                response_deserializer=example__implementation__pb2.Status.FromString,
                )
        self.submit_model = channel.unary_unary(
                '/example_implementation.FLServer/submit_model',
                request_serializer=example__implementation__pb2.ModelSubmission.SerializeToString,
                response_deserializer=example__implementation__pb2.Status.FromString,
                )
        self.submit_importances = channel.unary_unary(
                '/example_implementation.FLServer/submit_importances',
                request_serializer=example__implementation__pb2.ImportancesSubmission.SerializeToString,
                response_deserializer=example__implementation__pb2.Status.FromString,
                )
        self.get_model_info = channel.unary_unary(
                '/example_implementation.FLServer/get_model_info',
                request_serializer=example__implementation__pb2.ModelsFetchRequest.SerializeToString,
                response_deserializer=example__implementation__pb2.ModelInfoReply.FromString,
                )
        self.get_models = channel.unary_unary(
                '/example_implementation.FLServer/get_models',
                request_serializer=example__implementation__pb2.ModelsFetchRequest.SerializeToString,
                response_deserializer=example__implementation__pb2.ModelsFetchReply.FromString,
                )
        self.get_importances = channel.unary_unary(
                '/example_implementation.FLServer/get_importances',
                request_serializer=example__implementation__pb2.Empty.SerializeToString,
                response_deserializer=example__implementation__pb2.Importances.FromString,
                )
        self.clear = channel.unary_unary(
                '/example_implementation.FLServer/clear',
                request_serializer=example__implementation__pb2.Empty.SerializeToString,
                response_deserializer=example__implementation__pb2.Status.FromString,
                )


class FLServerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def request_admission(self, request, context):
        """Island admin
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def submit_model(self, request, context):
        """Submit a model
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def submit_importances(self, request, context):
        """Submit importances
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def get_model_info(self, request, context):
        """Fetch info on models
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def get_models(self, request, context):
        """Fetch models
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def get_importances(self, request, context):
        """Fetch importances
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def clear(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FLServerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'request_admission': grpc.unary_unary_rpc_method_handler(
                    servicer.request_admission,
                    request_deserializer=example__implementation__pb2.AddmissionRequest.FromString,
                    response_serializer=example__implementation__pb2.Status.SerializeToString,
            ),
            'submit_model': grpc.unary_unary_rpc_method_handler(
                    servicer.submit_model,
                    request_deserializer=example__implementation__pb2.ModelSubmission.FromString,
                    response_serializer=example__implementation__pb2.Status.SerializeToString,
            ),
            'submit_importances': grpc.unary_unary_rpc_method_handler(
                    servicer.submit_importances,
                    request_deserializer=example__implementation__pb2.ImportancesSubmission.FromString,
                    response_serializer=example__implementation__pb2.Status.SerializeToString,
            ),
            'get_model_info': grpc.unary_unary_rpc_method_handler(
                    servicer.get_model_info,
                    request_deserializer=example__implementation__pb2.ModelsFetchRequest.FromString,
                    response_serializer=example__implementation__pb2.ModelInfoReply.SerializeToString,
            ),
            'get_models': grpc.unary_unary_rpc_method_handler(
                    servicer.get_models,
                    request_deserializer=example__implementation__pb2.ModelsFetchRequest.FromString,
                    response_serializer=example__implementation__pb2.ModelsFetchReply.SerializeToString,
            ),
            'get_importances': grpc.unary_unary_rpc_method_handler(
                    servicer.get_importances,
                    request_deserializer=example__implementation__pb2.Empty.FromString,
                    response_serializer=example__implementation__pb2.Importances.SerializeToString,
            ),
            'clear': grpc.unary_unary_rpc_method_handler(
                    servicer.clear,
                    request_deserializer=example__implementation__pb2.Empty.FromString,
                    response_serializer=example__implementation__pb2.Status.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'example_implementation.FLServer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class FLServer(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def request_admission(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/example_implementation.FLServer/request_admission',
            example__implementation__pb2.AddmissionRequest.SerializeToString,
            example__implementation__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def submit_model(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/example_implementation.FLServer/submit_model',
            example__implementation__pb2.ModelSubmission.SerializeToString,
            example__implementation__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def submit_importances(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/example_implementation.FLServer/submit_importances',
            example__implementation__pb2.ImportancesSubmission.SerializeToString,
            example__implementation__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def get_model_info(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/example_implementation.FLServer/get_model_info',
            example__implementation__pb2.ModelsFetchRequest.SerializeToString,
            example__implementation__pb2.ModelInfoReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def get_models(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/example_implementation.FLServer/get_models',
            example__implementation__pb2.ModelsFetchRequest.SerializeToString,
            example__implementation__pb2.ModelsFetchReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def get_importances(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/example_implementation.FLServer/get_importances',
            example__implementation__pb2.Empty.SerializeToString,
            example__implementation__pb2.Importances.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def clear(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/example_implementation.FLServer/clear',
            example__implementation__pb2.Empty.SerializeToString,
            example__implementation__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
