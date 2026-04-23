# P&S Modern SSDs

# Basics of NAND Flash-Based SSDs

Dr. Mohammad Sadrosadati

Prof. Onur Mutlu

ETH Zürich

Fall 2022

12 October 2022

 SSD Organization & Request Handling   
NAND Flash Organization

 A modern SSD is a complicated system that consists of multiple cores, HW controllers, DRAM, and NAND flash memory packages

![](images/156e8e5dca7f188b0ddde85761cf979f161f290dbebe8a916e3a8f8f21ce3772.jpg)

<details>
<summary>text_image</summary>

SSD Controller
Core Core Core
HW
Flash Ctrl.
Request Handler
ECC/Randomizer
Encryption Engine
0.001 × 1,024 = 1 GB
8 × 128 GB = 1 TB
LPDDR DRAM
NAND Packages
</details>

Samsung PM853T 960GB Enterprise SSD (from https://www.tweaktown.com/reviews/6695/samsung-pm853t-960gb-enterprise-ssd-review/index.html)

# Another Overview

# Host Interface Layer (HIL)

# Flash Translation Layer (FTL)

Data Cache Management

Address Translation

GC/WL/Refresh/…

# Flash Controller

ECC

Randomizer

CTR

CTR

NAND

Flash

Package

NAND

Flash

Package

NAND

Flash

Package

# DRAM

Host Request Queue

Write Buffer

Logical-to-Physical Mappings

Metadata (e.g., P/E Cycles)

![](images/e6f2bde518463f973976d192f78ecfbcf1e94fd75ef306412e5f66d895050355.jpg)

<details>
<summary>text_image</summary>

SAMSUNG 416
K90KGY8S7M-CKD0
HPCE459H
SAMSUNG 416
K90KGY8S7M-CKD0
HPCE459H
SAMSUNG 416
K90KGY8S7M-CKD0
HPCE459H
SAMSUNG 416
K90KGY8S7M-CKD0
HPCE459H
100
C44
100
C44
100
C44
100
C44
100
C44
100
C44
100
C44
100
C44
100
C44
100
C44
100
C44
100
C44
100
C44
102
C44
100
C44
100
C44
100
C44
100
C44
100
C44
100
C44
100
C44
100
C44
100
C44
100
C44
100
C44
100
C43
</details>

# Host Interface Layer (HIL)

Flash Translation Layer (FTL)

Data Cache Management

Address Translation

GC/WL/Refresh/…

![](images/d9b87a16d784c651ef577cb485387250db8a4694f519914cad052c38c681f04c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Flash Controller"] --> B["ECC"]
    A --> C["Randomizer"]
    A --> D["CTRL"]
    D --> E["NAND Flash Package"]
    A --> F["NAND Flash Package"]
    A --> G["..."]
    A --> H["NAND Flash Package"]
    H --> I["NAND Flash Package"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#ccf,stroke:#333
    style D fill:#cfc,stroke:#333
    style E fill:#ffc,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#ffc,stroke:#333
    style H fill:#ffc,stroke:#333
```
</details>

# DRAM

Host Request Queue

Write Buffer

Logical-to-Physical Mappings

Metadata (e.g., P/E Cycles)

 Communication with the host operating system (receives & returns requests)

Via a certain interface (SATA or NVMe)

A host I/O request includes

Request direction (read or write)

 Offset (start sector address)

 Size (number of sectors)

Typically aligned by 4 KiB

# Request Handling: Write

Host Interface Layer (HIL)

# Flash Translation Layer (FTL)

Data Cache Management

Address Translation

GC/WL/Refresh/…

![](images/dc507aa04f0298bf6fe60b68a61731ff9a30c2a610d57ade5e4efea62b58fd6b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Flash Controller"] --> B["ECC"]
    A --> C["Randomizer"]
    A --> D["..."]
    A --> E["CTRL"]
    E --> F["NAND Flash Package"]
    A --> G["NAND Flash Package"]
    G --> H["..."]
    G --> I["NAND Flash Package"]
    I --> J["NAND Flash Package"]
```
</details>

# DRAM

Host Request Queue

Write Buffer

Logical-to-Physical Mappings

Metadata (e.g., P/E Cycles)

 Buffering data to write (read from NAND flash memory)

Essential to reducing write latency   
Enables flexible I/O scheduling   
Helpful for improving lifetime (not so likely)

Limited size (e.g., tens of MBs)

 Needs to ensure data integrity even under sudden power-off   
Most DRAM capacity is used for L2P mappings

# Request Handling: Write

Host Interface Layer (HIL)

# Flash Translation Layer (FTL)

Data Cache Management

Address Translation

GC/WL/Refresh/…

![](images/df6f9037593e826761a475f2206dac403f08edd316aa82cea7bc7ffbdaccf787.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Flash Controller"] --> B["ECC"]
    A --> C["Randomizer"]
    A --> D["..."]
    A --> E["CTRL"]
    E --> F["NAND Flash Package"]
    A --> G["NAND Flash Package"]
    G --> H["..."]
    G --> I["NAND Flash Package"]
    I --> J["NAND Flash Package"]
```
</details>

# DRAM

Host Request Queue

Write Buffer

Logical-to-Physical Mappings

Metadata (e.g., P/E Cycles)

Core functionality for out-of-place writes

 To hide the erase-before-write property

 Needs to maintain L2P mappings

Logical Page Address (LPA)   
 Physical Page Address (PPA)

 Mapping granularity: 4 KiB

 4 Bytes for 4 KiB  0.1% of SSD capacity

Host Interface Layer (HIL)

# Flash Translation Layer (FTL)

Data Cache Management

Address Translation

GC/WL/Refresh/…

![](images/8d3513864b6775024b4482902f549571137b6422d1206e5e6f315a9bb626ffe5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Flash Controller"] --> B["ECC"]
    A --> C["Randomizer"]
    A --> D["..."]
    A --> E["CTRL"]
    E --> F["NAND Flash Package"]
    A --> G["NAND Flash Package"]
    G --> H["..."]
    A --> I["NAND Flash Package"]
    I --> J["NAND Flash Package"]
```
</details>

# DRAM

Host Request Queue

Write Buffer

Logical-to-Physical Mappings

Metadata (e.g., P/E Cycles)

Garbage collection (GC)

 Reclaims free pages   
Selects a victim block  copies all valid pages  erase the victim block

Wear-leveling (WL)

 Evenly distributes P/E cycles across NAND flash blocks   
Hot/cold swapping

Data refresh

 Refresh pages with long retention ages

# Request Handling: Write

Host Interface Layer (HIL)

Flash Translation Layer (FTL)

Data Cache Management

Address Translation

GC/WL/Refresh/…

![](images/3fdf9503373cbfd9621dda1e6d484b3a89bfcaf1ac58be39356c28872b9195b4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Flash Controller"] --> B["ECC"]
    A --> C["Randomizer"]
    A --> D["CTRL"]
    D --> E["NAND Flash Package"]
    A --> F["NAND Flash Package"]
    F --> G["..."]
    A --> H["NAND Flash Package"]
    H --> I["NAND Flash Package"]
    A --> J["..."]
    A --> K["CTRL"]
```
</details>

# DRAM

Host Request Queue

Write Buffer

Logical-to-Physical Mappings

Metadata (e.g., P/E Cycles)

# Randomizer

Scrambling data to write   
To avoid worst-case data patterns that can lead to significant errors

# Error-correcting codes (ECC)

Can detect/correct errors: e.g., 72 bits/1 KiB error-correction capability   
Stores additional parity information together with raw data

# Issues NAND flash commands

# Request Handling: Read

Host Interface Layer (HIL)

# Flash Translation Layer (FTL)

Data Cache Management

Address Translation

GC/WL/Refresh/…

![](images/9ea43e4d6819cde36166e51a26fdfed7e715ec632463bef2c1bfd5666a744462.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Flash Controller"] --> B["ECC"]
    A --> C["Randomizer"]
    A --> D["..."]
    A --> E["CTRL"]
    E --> F["NAND Flash Package"]
    A --> G["NAND Flash Package"]
    G --> H["..."]
    G --> I["NAND Flash Package"]
    I --> J["NAND Flash Package"]
```
</details>

# DRAM

Host Request Queue

Write Buffer

Logical-to-Physical Mappings

Metadata (e.g., P/E Cycles)

 First checks if the request data exists in the write buffer

 If so, returns the corresponding request immediately with the data

 A host read request can be involved with several pages

Such a request can be returned only after all the requested data is ready

# Request Handling: Read

Host Interface Layer (HIL)

# Flash Translation Layer (FTL)

Data Cache Management

Address Translation

GC/WL/Refresh/…

![](images/a9de280c60419696b051a33947aefeaf09e5589a1bfa99211dc729505b4fb473.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Flash Controller"] --> B["ECC"]
    A --> C["Randomizer"]
    A --> D["..."]
    A --> E["CTRL"]
    E --> F["NAND Flash Package"]
    A --> G["NAND Flash Package"]
    G --> H["..."]
    G --> I["NAND Flash Package"]
    I --> J["NAND Flash Package"]
```
</details>

# DRAM

Host Request Queue

Write Buffer

Logical-to-Physical Mappings

Metadata (e.g., P/E Cycles)

 Finds the PPA where the request data is stored from the L2P mapping table

# Request Handling: Read

Host Interface Layer (HIL)

Flash Translation Layer (FTL)

Data Cache Management

Address Translation

GC/WL/Refresh/…

![](images/a6503a1d7050f7dff4c572bbcdd5ee995c5538682bcd5d170edba2c8e2e65cc0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Flash Controller"] --> B["ECC"]
    A --> C["Randomizer"]
    A --> D["CTRL"]
    D --> E["NAND Flash Package"]
    D --> F["NAND Flash Package"]
    D --> G["NAND Flash Package"]
    D --> H["..."]
    D --> I["CTRL"]
```
</details>

# DRAM

Host Request Queue

Write Buffer

Logical-to-Physical Mappings

Metadata (e.g., P/E Cycles)

 First reads the raw data from the flash chip   
 Performs ECC decoding   
 Derandomizes the raw data   
 ECC decoding can fail

Retries reading of the page w/ adjusted $V _ { \tt R E F }$

 SSD Organization & Request Handling   
NAND Flash Organization

Basically, it is a transistor

![](images/e98f0fbd42fc7d7d9d0fd8f4e9b83c9407f67cba6edb819d910dc5efb5f782ff.jpg)

<details>
<summary>text_image</summary>

G
(Control Gate)
S
(Source)
Substrate
I_D
D
(Drain)
</details>

![](images/b00be20e67da3bc3f807f1817e5402511d7d9a8f83c0b8e28d651f054916c13c.jpg)

<details>
<summary>line</summary>

| V_GS Range       | I_D Value |
| ---------------- | --------- |
| V_GS < V_TH      | Low       |
| V_TH             | High      |
| V_GS > V_TH      | High      |
</details>

(Threshold Voltage)

# A Flash Cell

Basically, it is a transistor

 $\boldsymbol { \mathsf { W } } /$ a special material: Floating gate (2D) or Charge trap (3D)

![](images/9622722e70e665e4880c78ce7eb9f85b77d6b42734bb78d2aa3823b2b9e0e8e0.jpg)

<details>
<summary>text_image</summary>

G
(Control Gate)
FG
(Floating Gate)
S
(Source)
Substrate
D
(Drain)
</details>

![](images/f1f7b3ff78e7af9b8c973a0e689c07c535cc22bda1f512093d98d4487dbf533b.jpg)

<details>
<summary>line</summary>

| V_GS | I_D |
|------|-----|
| V_TH | Peak |
</details>

Basically, it is a transistor

 w/ a special material: Floating gate (2D) or Charge trap (3D)   
Can hold electrons in a non-volatile manner

![](images/231d97208de674cfa567935eeead10143a74fec63249fd11150973366c6451ba.jpg)

<details>
<summary>text_image</summary>

VPGM = 20 V
G
(Control Gate)
FG
(Floating Gate)
Tunneling
S
(Source)
GND
Substrate
D
(Drain)
</details>

![](images/8c5ebf5ff5a696d13b6bd7a8dfe84bedaecb92f8604a4197fcb50b4f097b374a.jpg)

<details>
<summary>line</summary>

| V_GS | I_D |
|------|-----|
| V_TH | Peak |
</details>

Basically, it is a transistor

 $\boldsymbol { \mathsf { W } } /$ a special material: Floating gate (2D) or Charge trap (3D)   
Can hold electrons in a non-volatile manner   
Changes the cell’s threshold voltage $( V _ { \mathsf { T H } } )$

![](images/b28ab9b48af002ee4605a390e069235536d26186f74581f3ca6ddedea51ef3c4.jpg)

<details>
<summary>text_image</summary>

GND
G
(Control Gate)
FG
(Floating Gate)
Tunneling
20 V
Substrate
S (Source) D (Drain)
</details>

![](images/e1e28a6a97dbed9b9dbc8aafeff2853f5ec2ee6ae416b3f66353b7d16800a2a0.jpg)

<details>
<summary>line</summary>

| V_TH | I_D (Blue Line) | I_D (Red Line) |
|------|-----------------|----------------|
| V_TH | ~0              | ~0             |
| V_TH' | High            | Low            |
| V_GS | High            | High           |
</details>

Multi-leveling: A flash cell can store multiple bits

Program: Inject electrons   
![](images/028213a685cfa7f30f759d413ff536073432bdea01c926994882ef4dbe1e4b41.jpg)

<details>
<summary>natural_image</summary>

Simple diagram with a circle and two arrows pointing upward and downward (no text or symbols)
</details>

Flash Cell   
Erase: Eject electrons

![](images/23f55fc2ab34ad2affb062afa73e9e1d22c7d4631423134456334df3532da959.jpg)

<details>
<summary>text_image</summary>

10
00
01
11
</details>

2-bit Multi-level Cell

Retention loss: A cell leaks electrons over time

![](images/888d98f5fa461a20887e766724c6e1f35eac6b2c6fe172c9993bc5b9dbfa4a84.jpg)

<details>
<summary>text_image</summary>

10
00
01
11
</details>

![](images/e7eb8e55d273fa54a12e4d927dd25b28645b7c410122299c346757253f3ee3cf.jpg)

![](images/5f90f8dfeca63fa896d7a3f3e58935d2ede0e6161cc8d01be7b3ad2314a447ec.jpg)

<details>
<summary>text_image</summary>

10
00
01
11
</details>

![](images/8556a18578900303b630d5c66008b62ef394a38d651f0550ef87f2fdd9a13fd9.jpg)

![](images/6f1d8061a89ce7905cbbf80b23ef443af30117635ed64468e733f3ebff20a456.jpg)

<details>
<summary>pie</summary>

| Category | Value |
|---|---|
| 00 | 10 |
| 01 | 01 |
| 10 | 11 |
The chart displays a single filled circle representing 10 units of a total. The other three columns are not explicitly labeled but are visually represented as smaller segments on the right side.
</details>

Retention error!

Limited lifetime: A cell wears out after P/E cycling

![](images/16d01b8f36c7d6d35bb4e390a0f570da212e7ce535ffb38e3efd9ee2ca3763f7.jpg)

<details>
<summary>text_image</summary>

10
00
01
11
</details>

1 year @ 1K P/E cycles   
![](images/f1d5095b1bd996f1eb3eefa5a89496c9b15b9733f6474006d0cb4d4256932468.jpg)

![](images/c7baa61b76fcf96ab0183f6febc161c762eed949a021856a6116a732682fa121.jpg)

<details>
<summary>pie</summary>

| Category | Value |
|---|---|
| 10 | 10 |
| 00 | 00 |
| 01 | 01 |
| 11 | 11 |
</details>

![](images/5ed1acafa120489c2042530c4a3e37590c34bded31afb540b6dd2dbeb7ec9bb7.jpg)  
1 year @ 10K P/E cycles

![](images/1e8740a7fdccc89e142aee9cd99ea613d1d70f48c1274323d8f272ea20caee13.jpg)

<details>
<summary>pie</summary>

| Segment | Value |
|---|---|
| 10 | 10 |
| 00 | 00 |
| 01 | 01 |
| 11 | 11 |
</details>

Retention error!

 Multiple (e.g., 128) flash cells are serially connected

![](images/9a386de25e5f4957f9fc1bde8a3e4239f393ddbcc5d65b1631891c6b1b7b5c79.jpg)

<details>
<summary>text_image</summary>

Bitline (BL)
V_PASS (> 6V)
Target Cell
V_PASS
V_PASS
NAND String
Bitline (BL)
</details>

# Pages and Blocks

 A large number $( > 1 0 0 , 0 0 0 )$ of cells operate concurrently

![](images/5b171eb909305f06df1649ff0dfa8128c3f8236c47d965dc57ed42e3c50a6703.jpg)

<details>
<summary>other</summary>

| Block | Wordline WL0 | Wordline WL1 | Wordline WL126 | Wordline WL127 |
|-------|--------------|--------------|----------------|----------------|
| Block | 16 + α KiB    | -            | -              | -              |
| Block = {(# of WL) × (# of bits per cell)} pages | -            | -            | -              | -              |
</details>

# Pages and Blocks (Continued)

Program and erase: Unidirectional

Programming a cell  Increasing the cell’s $\mathsf { V } _ { \mathsf { T H } }$   
Eraseing a cell  Decreasing the cell’s $\mathsf { V } _ { \mathsf { T H } }$

 Programming a page cannot change ‘0’ cells to ‘1’ cells  Erase-before-write property

Erase unit: Block

Increase erase bandwidth   
Makes in-place write on a page very inefficient  Out-of-place write & GC

![](images/6db35aafbbb4fa41c3cf39317546ac186099bcf8adccab0f23f8b244c89b01a3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph_Block["Block = {(# of WL) × (# of bits per cell)} pages"]
        direction TB
        A["Wordline WL0"] --> B["Block"]
        C["Wordline WL1"] --> D["Block"]
        E["Wordline WL126"] --> F["Block"]
        G["Wordline WL127"] --> H["Block"]
    end
    subgraph_Page["Page = 16 + α KiB"]
        I["Block"]
        J["Wordline"]
        K["Wordline"]
        L["Wordline"]
        M["Wordline"]
        N["Wordline"]
    end
    style Block fill:#f9f,stroke:#333
    style Page fill:#ccf,stroke:#333
    note right of A: BL0
    note left of I: BL1
    note right of J: BL2
    note right of K: BL3
    note left of L: BL126
    note right of M: BL127
    note right of N: BL132,095
```
</details>

 A large number (> 1,000) of blocks share bitlines in a plane

![](images/c77682faff06a8357f7f0cdef86df83e3d6aafef1e2054cdfbeaedc5b938fb3b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph BL0
        WL0["WL₀"] --> BL0
        WL1["WL₁"] --> BL0
        WL126["WL₁₂₆"] --> BL0
        WL127["WL₁₂₇"] --> BL0
    end
    
    subgraph BL1
        Block0["Block₀"] --> BL0
        Block1["Block₁"] --> BL0
        Block2["Block₂"] --> BL0
        Block2_047["Block₂,₀₄₇"] --> BL0
        Block0 --> BL1["BL₁"]
        Block1 --> BL1
        Block2 --> BL1
        Block2_047 --> BL1
        Block0 --> BL2["BL₂"]
        Block1 --> BL2
        Block2 --> BL2
        Block2_047 --> BL2
        Block0 --> BL3["BL₃"]
        Block1 --> BL3
        Block2 --> BL3
        Block2_047 --> BL3
    end
    
    subgraph BL132_095
        Block0_Block1_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2
    end
    
    subgraph Block0
        Block0_Block1_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block3
    end
    
    subgraph Block1
        Block1_Block1_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block2_Block3
    end
    
    subgraph Block2
        Block1_Block1_Block2_Block3
        Block1_Block1_Block3
        Block1_Block1_Block4
        Block1_Block1_Block5
        Block1_Block1_Block6
        Block1_Block1_Block7
        Block1_Block1_Block8
        Block1_Block1_Block9
        Block1_Block1_Block10
        Block1_Block1_Block11
        Block1_Block1_Block12
        Block1_Block1_Block13
        Block1_Block1_Block14
        Block1_Block1_Block15
        Block1_Block1_Block16
        Block1_Block1_Block17
        Block1_Block1_Block18
        Block1_Block1_Block19
        Block1_Block1  Block13
    end
```
</details>

 A large number (> 1,000) of blocks share bitlines in a plane

![](images/d5b71dacd8f7b1eb282f05712cc4d8fe2417aafc1ac0446c0658cfcdd175fba4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph "String Select Line"
        A["SSL"] --> B["WL0"]
        B --> C["WL1"]
        C --> D["WL126"]
        D --> E["WL127"]
        E --> F["GSL"]
    end

    subgraph "Ground Select Line"
        G["Block0"] --> H["Block1"]
        H --> I["Block2"]
        I --> J["..."]
        K["Block2,047"]
    end

    A --> G
    B --> H
    C --> I
    D --> J
    E --> K
    F --> L["Block0"]
    G --> M["Block1"]
    H --> N["Block2"]
    I --> O["..."]
    J --> P["Block2,047"]
    K --> Q["Block0"]
    L --> R["Block1"]
    M --> S["Block2"]
    N --> T["..."]
    O --> U["Block2,047"]
    P --> V["Block0"]
    Q --> W["Block1"]
    R --> X["Block2"]
    S --> Y["..."]
    T --> Z["Block2,047"]
    U --> AA["Block0"]
    V --> AB["Block1"]
    W --> AC["Block2"]
    X --> AD["..."]
    Y --> AE["Block2,047"]
    Z --> AF["Block0"]
    AA --> AG["Block1"]
    AB --> AH["Block2"]
    AC --> AI["..."]
    AD --> AJ["Block2,047"]
    AE --> AK["Block0"]
    AF --> AL["Block1"]
    AG --> AM["Block2"]
    AH --> AN["..."]
    AI --> AO["Block2,047"]
    AJ --> AP["Block0"]
    AK --> AQ["Block1"]
    AL --> AR["Block2"]
    AM --> AS["..."]
    AN --> AT["Block2,047"]
```
</details>

# Planes and Dies

 A die contains multiple (e.g., 2 – 4) planes

![](images/19f82f6a95e19cc7c92c249a091ea2d98c9ae4ced5c15d0547a41fc391b668c8.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Block0
        A["Block₀"] --> B["Block₁"]
        C["Block₁"] --> D["..."]
        E["Block₂"] --> F["..."]
        G["..."]
    end
    subgraph Block2
        H["Block₂"] --> I["..."]
        J["..."]
    end
    subgraph Block2,047
        K["Block₂,047"] --> L["..."]
    end

    subgraph Row/Column Decoders
        M["Plane₀"] --> N["Plane₁"] --> O["Plane₂"] --> P["Plane₃"]
        Q["Page Buffers"] --> R["Peripheral Circuits"]
    end

    style Block0 fill:#f9f,stroke:#333
    style Block2 fill:#f9f,stroke:#333
    style Block2,047 fill:#f9f,stroke:#333

    note right of M: Planes share decoders: limits internal parallelism (only operations @ the same WL offset)

    note right of A: A 21-nm 2D NAND Flash Die
    note right of A: Planes share decoders: limits internal parallelism (only operations @ the same WL offset)
```
</details>

# P&S Modern SSDs

# Basics of NAND Flash-Based SSDs

Dr. Mohammad Sadrosadati

Prof. Onur Mutlu

ETH Zürich

Fall 2022

12 October 2022