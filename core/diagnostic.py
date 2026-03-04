from stdatalog_core.HSD.HSDatalog import HSDatalog
from pathlib import Path

acq_dir = "/Users/kgun/Downloads/Sensor_STWIN/vel-fissa/OK/PMS_50rpm"  # e.g. .../vel-fissa/OK/PMS_50rpm/STWIN_00001


hsd = HSDatalog()
_ = hsd.validate_hsd_folder(acq_dir)
hsd_instance = hsd.create_hsd(acquisition_folder=acq_dir)

for name in ["hts221_temp", "hts221_hum", "lps22hh_press", "lps22hh_temp", "iis3dwb_acc", "iis3dwb_gyro"]:
    try:
        sensor = hsd.get_sensor(hsd_instance, name)
        df_obj = hsd.get_dataframe(hsd_instance, sensor)
        print(f"\n--- {name} ---")
        print(f"  type: {type(df_obj)}")
        if isinstance(df_obj, list):
            print(f"  len: {len(df_obj)}")
            for i, item in enumerate(df_obj[:3]):  # first 3 items
                print(f"  [{i}] type={type(item).__name__}", end="")
                if hasattr(item, 'shape'):
                    print(f", shape={item.shape}", end="")
                elif hasattr(item, '__len__'):
                    print(f", len={len(item)}", end="")
                print(f", value preview={str(item)[:80]}")
    except Exception as e:
        print(f"\n--- {name} EXCEPTION: {e}")



"""
    Usage & Output:
    (.venv) (base) kgun@Host-004 Systems-Project % python core/diagnostic.py

--- hts221_temp ---
  type: <class 'list'>
  len: 2
  [0] type=DataFrame, shape=(850, 2), value preview=          Time  TEMP [Celsius]
0     0.206115       32.213997
1     0.285572    
  [1] type=DataFrame, shape=(25, 2), value preview=         Time  TEMP [Celsius]
0   67.779513       32.806190
1   67.859012       

--- hts221_hum ---
  type: <class 'list'>
  len: 2
  [0] type=DataFrame, shape=(850, 2), value preview=          Time  TEMP [Celsius]
0     0.206115       32.213997
1     0.285572    
  [1] type=DataFrame, shape=(25, 2), value preview=         Time  TEMP [Celsius]
0   67.779513       32.806190
1   67.859012       

--- lps22hh_press ---
  type: <class 'list'>
  len: 1
  [0] type=DataFrame, shape=(14200, 2), value preview=            Time  PRESS [hPa]
0       0.668811  1013.406494
1       0.673783  10

--- lps22hh_temp ---
  type: <class 'list'>
  len: 1
  [0] type=DataFrame, shape=(14200, 2), value preview=            Time  PRESS [hPa]
0       0.668811  1013.406494
1       0.673783  10

--- iis3dwb_acc ---
  type: <class 'list'>
  len: 2
  [0] type=DataFrame, shape=(1924000, 4), value preview=              Time   A_x [g]   A_y [g]   A_z [g]
0         0.172246  0.033672 -0
  [1] type=DataFrame, shape=(1000, 4), value preview=          Time   A_x [g]   A_y [g]   A_z [g]
0    72.748271  0.008784 -0.107360 

--- iis3dwb_gyro ---
  type: <class 'list'>
  len: 2
  [0] type=DataFrame, shape=(1924000, 4), value preview=              Time   A_x [g]   A_y [g]   A_z [g]
0         0.172246  0.033672 -0
  [1] type=DataFrame, shape=(1000, 4), value preview=          Time   A_x [g]   A_y [g]   A_z [g]
0    72.748271  0.008784 -0.107360 
       
 """