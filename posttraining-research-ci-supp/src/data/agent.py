from dataclasses import dataclass, field, asdict
from typing import List, Dict, Literal
from dacite import from_dict


@dataclass
class Segment:
    start_index: int
    end_index: int
    annotations: List[Dict[str, str]]


@dataclass
class Turn:
    role: Literal["assistant", "user", "system", "service"]
    content: str
    segments: List[Segment] = field(default_factory=list)

    def segment_text(self, segment: Segment) -> str:
        return self.content[segment.start_index:segment.end_index]
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Turn":
        return from_dict(data_class=cls, data=data)
    
    def as_dict(self) -> Dict:
        return asdict(self)
    
    def overlaps_segment(self, segment: Segment) -> bool:
        return any(
            seg.start_index <= segment.end_index and seg.end_index >= segment.start_index
            for seg in self.segments
        )
    
    def add_segment(self, segment: Segment) -> None:
        if self.overlaps_segment(segment):
            raise ValueError("Segment overlaps with existing segments.")
        self.segments.append(segment)

    def as_pretty_str(self, highlight_annotations: bool = True) -> str:
        offset = 0
        text = self.content
        for segment in self.segments:
            start_index = segment.start_index + offset
            end_index = segment.end_index + offset
            if highlight_annotations:
                highlight_start = "[bold red]"
                highlight_end = "[/bold red]"
            else:
                highlight_start = ""
                highlight_end = ""
            text = text[:start_index] + highlight_start + text[start_index:end_index] + highlight_end + text[end_index:]
            # Adjust subsequent segments' indices
            offset += len(highlight_start) + len(highlight_end)
        return f"{self.role.upper()}: {text}"


@dataclass
class Request:
    required_attributes: Dict[str, str]
    not_required_attributes: Dict[str, str]
    turns: List[Turn]

    @classmethod
    def from_dict(cls, data: Dict) -> "Request":
        return from_dict(data_class=cls, data=data)
    
    def as_dict(self) -> Dict:
        return asdict(self)
