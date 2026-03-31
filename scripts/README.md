# scripts/

The main runnable script in this directory is:

```bash
scripts/run_mit4sl.sh
```

This script automatically combines:

- the scenario-level config: `configs/<scenario>/protocol.yaml`
- the target config: `configs/<scenario>/<target>.yaml`

and then calls:

```bash
python src/train_MiT4SL.py --cfg <protocol> --cfg <target>
```

This document uses **A549 in the cross-cell-line setting** as the default example.  
Environment setup is described in the top-level project README; here we only explain how to use the script itself.

---

## 1. Simplest usage: the A549 example

First enter the project root:

```bash
cd /home/siyutao/SL/Github_project/MiT4SL_nature_final
```

Then run:

```bash
bash scripts/run_mit4sl.sh
```

By default, this is equivalent to:

```bash
bash scripts/run_mit4sl.sh \
  --config-dir cross_cell_line \
  --target A549
```

And that further expands to:

```bash
python src/train_MiT4SL.py \
  --cfg configs/cross_cell_line/protocol.yaml \
  --cfg configs/cross_cell_line/Multi_5_to_A549.yaml
```

If you need to override the configured runtime device, add `--device <id>` (for example, `--device 0`).

If you want to inspect the resolved command before actually starting training, run:

```bash
bash scripts/run_mit4sl.sh --dry-run
```

---

## 2. What this script does for you

`run_mit4sl.sh` mainly does four things:

1. locates the repository root
2. finds `protocol.yaml` under the selected `--config-dir`
3. resolves the target config from `--target`
4. launches `src/train_MiT4SL.py`

So in practice, you do not need to manually assemble the full training command every time.

---

## 3. Argument reference

### `--config-dir`

Specifies the scenario directory under `configs/`.

For example:

```bash
bash scripts/run_mit4sl.sh --config-dir cross_cell_line
```

It can also be switched to other scenarios, for example:

```bash
bash scripts/run_mit4sl.sh --config-dir cell_line_specific/random
```

```bash
bash scripts/run_mit4sl.sh --config-dir recom_sl_partner
```

---

### `--target`

Specifies the target config within the selected scenario.

Using A549 as an example, you can write:

```bash
bash scripts/run_mit4sl.sh --config-dir cross_cell_line --target A549
```

You can also use the full target name:

```bash
bash scripts/run_mit4sl.sh --config-dir cross_cell_line --target Multi_5_to_A549
```

or the exact yaml filename:

```bash
bash scripts/run_mit4sl.sh --config-dir cross_cell_line --target Multi_5_to_A549.yaml
```

The script will automatically resolve the correct target config.

---

### `--dry-run`

Prints the resolved command without actually starting training.

```bash
bash scripts/run_mit4sl.sh --dry-run
```

This is recommended the first time you try a new scenario or a new cell line.

---

### `--list-targets`

Lists all available targets in a scenario directory.

For example:

```bash
bash scripts/run_mit4sl.sh --config-dir cross_cell_line --list-targets
```

---

### `--save-model-path`

Manually specifies the output directory.

For example:

```bash
bash scripts/run_mit4sl.sh --save-model-path result/a549_run
```

---

### `--device`

Overrides the runtime device without editing the selected `protocol.yaml`.

For example:

```bash
bash scripts/run_mit4sl.sh --device 0
```

The value is forwarded to `src/train_MiT4SL.py --device ...`. You can use GPU indices such as `0`, or explicit device strings such as `cpu` and `cuda:1`.

---

### `--python`

Manually specifies the Python interpreter.

For example:

```bash
bash scripts/run_mit4sl.sh --python /path/to/python
```

If your environment is already configured as described in the main project README, you usually do not need to set this explicitly.

---

## 4. The two key ideas for readers: both scenario and cell line are replaceable

This script is not limited to A549.  
A549 is only used here as the default example. In practice, you can replace:

- the **scenario** via `--config-dir`
- the **cell line / target** via `--target`

For example:

### 4.1 Replace A549 with another cell line

```bash
bash scripts/run_mit4sl.sh --config-dir cross_cell_line --target Jurkat
```

or:

```bash
bash scripts/run_mit4sl.sh --config-dir cross_cell_line --target MeWo
```

### 4.2 Replace `cross_cell_line` with another scenario

```bash
bash scripts/run_mit4sl.sh --config-dir cell_line_specific/random --target A549
```

```bash
bash scripts/run_mit4sl.sh --config-dir cell_line_specific/cold_start --target A549
```

### 4.3 Switch to another task configuration

```bash
bash scripts/run_mit4sl.sh --config-dir recom_sl_partner --target A549_KRAS
```

If you are not sure which targets exist in a scenario, check them first with:

```bash
bash scripts/run_mit4sl.sh --config-dir <scenario> --list-targets
```

---

## 5. Quick command reference

### Default A549 run

```bash
bash scripts/run_mit4sl.sh
```

### Default A549 dry-run

```bash
bash scripts/run_mit4sl.sh --dry-run
```

### List all targets in the `cross_cell_line` scenario

```bash
bash scripts/run_mit4sl.sh --config-dir cross_cell_line --list-targets
```

### Run A549 in the `cell_line_specific/random` scenario

```bash
bash scripts/run_mit4sl.sh --config-dir cell_line_specific/random --target A549
```

### Run A549_KRAS in the `recom_sl_partner` scenario

```bash
bash scripts/run_mit4sl.sh --config-dir recom_sl_partner --target A549_KRAS
```

---

## 6. Related files

- script: `scripts/run_mit4sl.sh`
- training entrypoint: `src/train_MiT4SL.py`
- config directory: `configs/`
- config notes: `configs/README.md`
