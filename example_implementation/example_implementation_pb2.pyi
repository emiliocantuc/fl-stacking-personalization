from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AddmissionRequest(_message.Message):
    __slots__ = ["name"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class Importances(_message.Message):
    __slots__ = ["importances", "islands", "plain_importances"]
    IMPORTANCES_FIELD_NUMBER: _ClassVar[int]
    ISLANDS_FIELD_NUMBER: _ClassVar[int]
    PLAIN_IMPORTANCES_FIELD_NUMBER: _ClassVar[int]
    importances: _containers.RepeatedScalarFieldContainer[float]
    islands: _containers.RepeatedScalarFieldContainer[str]
    plain_importances: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, islands: _Optional[_Iterable[str]] = ..., importances: _Optional[_Iterable[float]] = ..., plain_importances: _Optional[_Iterable[float]] = ...) -> None: ...

class ImportancesSubmission(_message.Message):
    __slots__ = ["from_island", "importances"]
    FROM_ISLAND_FIELD_NUMBER: _ClassVar[int]
    IMPORTANCES_FIELD_NUMBER: _ClassVar[int]
    from_island: str
    importances: Importances
    def __init__(self, from_island: _Optional[str] = ..., importances: _Optional[_Union[Importances, _Mapping]] = ...) -> None: ...

class Model(_message.Message):
    __slots__ = ["model"]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    model: bytes
    def __init__(self, model: _Optional[bytes] = ...) -> None: ...

class ModelInfo(_message.Message):
    __slots__ = ["bytes", "from_island", "model_type", "trained_on_n"]
    BYTES_FIELD_NUMBER: _ClassVar[int]
    FROM_ISLAND_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    TRAINED_ON_N_FIELD_NUMBER: _ClassVar[int]
    bytes: int
    from_island: str
    model_type: str
    trained_on_n: int
    def __init__(self, trained_on_n: _Optional[int] = ..., bytes: _Optional[int] = ..., from_island: _Optional[str] = ..., model_type: _Optional[str] = ...) -> None: ...

class ModelInfoReply(_message.Message):
    __slots__ = ["info", "status"]
    INFO_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    info: _containers.RepeatedCompositeFieldContainer[ModelInfo]
    status: Status
    def __init__(self, status: _Optional[_Union[Status, _Mapping]] = ..., info: _Optional[_Iterable[_Union[ModelInfo, _Mapping]]] = ...) -> None: ...

class ModelSubmission(_message.Message):
    __slots__ = ["from_island", "info", "model"]
    FROM_ISLAND_FIELD_NUMBER: _ClassVar[int]
    INFO_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    from_island: str
    info: ModelInfo
    model: Model
    def __init__(self, from_island: _Optional[str] = ..., info: _Optional[_Union[ModelInfo, _Mapping]] = ..., model: _Optional[_Union[Model, _Mapping]] = ...) -> None: ...

class ModelsFetchReply(_message.Message):
    __slots__ = ["models", "models_info", "status"]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    MODELS_INFO_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedCompositeFieldContainer[Model]
    models_info: _containers.RepeatedCompositeFieldContainer[ModelInfo]
    status: Status
    def __init__(self, status: _Optional[_Union[Status, _Mapping]] = ..., models: _Optional[_Iterable[_Union[Model, _Mapping]]] = ..., models_info: _Optional[_Iterable[_Union[ModelInfo, _Mapping]]] = ...) -> None: ...

class ModelsFetchRequest(_message.Message):
    __slots__ = ["models"]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, models: _Optional[_Iterable[str]] = ...) -> None: ...

class Status(_message.Message):
    __slots__ = ["details", "success"]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    details: str
    success: bool
    def __init__(self, success: bool = ..., details: _Optional[str] = ...) -> None: ...
