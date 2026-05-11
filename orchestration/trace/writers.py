"""
Writers for step events (JSONL) and telemetry samples (Parquet/CSV).
"""

import json
from pathlib import Path
from typing import Optional, List

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

from .schema import StepEvent, TelemetrySample


class JSONLWriter:
    """
    JSONL writer for step events.
    
    Usage:
        writer = JSONLWriter("trace.jsonl")
        writer.write(event)
        writer.close()
    """
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w")
    
    def write(self, event: StepEvent):
        """Write one event."""
        line = json.dumps(event.to_dict())
        self._file.write(line + "\n")
    
    def flush(self):
        """Flush to disk."""
        self._file.flush()
    
    def close(self):
        """Close writer."""
        self._file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ParquetWriter:
    """
    Parquet writer for telemetry samples.
    
    Buffers samples in memory, writes in batches.
    Falls back to CSV if pyarrow not available.
    """
    
    def __init__(self, output_path: str, batch_size: int = 1000):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        
        self._buffer: List[TelemetrySample] = []
        self._writer: Optional[pq.ParquetWriter] = None
        self._schema: Optional[pa.Schema] = None
        
        if not PYARROW_AVAILABLE:
            print("Warning: pyarrow not available, falling back to CSV")
            # Fallback to CSV
            self._csv_fallback = CSVWriter(str(self.output_path).replace(".parquet", ".csv"))
    
    def write(self, sample: TelemetrySample):
        """Write one sample (buffered)."""
        if not PYARROW_AVAILABLE:
            self._csv_fallback.write(sample)
            return
        
        self._buffer.append(sample)
        
        if len(self._buffer) >= self.batch_size:
            self._flush()
    
    def _flush(self):
        """Flush buffer to Parquet file."""
        if not self._buffer:
            return
        
        # Convert to dict list
        data = [s.to_dict() for s in self._buffer]
        
        # Build pyarrow table
        table = pa.Table.from_pylist(data)
        
        # Initialize writer on first flush
        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(self.output_path, self._schema)
        
        # Write batch
        self._writer.write_table(table)
        
        self._buffer.clear()
    
    def close(self):
        """Flush and close writer."""
        if not PYARROW_AVAILABLE:
            self._csv_fallback.close()
            return
        
        self._flush()
        if self._writer:
            self._writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class CSVWriter:
    """
    CSV writer for telemetry samples (fallback when pyarrow unavailable).
    """
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w")
        self._header_written = False
    
    def write(self, sample: TelemetrySample):
        """Write one sample."""
        data = sample.to_dict()
        
        if not self._header_written:
            # Write header
            self._file.write(",".join(data.keys()) + "\n")
            self._header_written = True
        
        # Write row
        row = ",".join(str(v) if v is not None else "" for v in data.values())
        self._file.write(row + "\n")
    
    def flush(self):
        """Flush to disk."""
        self._file.flush()
    
    def close(self):
        """Close writer."""
        self._file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
