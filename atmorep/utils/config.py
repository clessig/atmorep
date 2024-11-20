import dataclasses as dc

@dc.dataclass
class FieldConfig:
    name: str # fields[x][0]
    m_lvls: list[int] # fields[x][2]
    token_shape: tuple[int] # fields[x][3] ()
    sample_shape: tuple[int] # (time, lat, lon)
    

