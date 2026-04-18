from pathlib import Path
import rawpy

root = Path("./rawref")
files = [ 
  "DSC07829.ARW",
  "DSC07830.ARW",
  "DSC07831.ARW"
]

for i, frame in enumerate(files):
  with rawpy.imread( str(root / frame)) as raw:
      cfa = raw.raw_image
      colors = raw.raw_colors
      corner = cfa[:64, :64]
      corner_colors = colors[:64, :64]
      
      print(f"image: {i}:")
      for ch in range(4):
          mask = corner_colors == ch
          print(f"channel {ch}: {corner[mask].mean():.3f}")  
      print("---")