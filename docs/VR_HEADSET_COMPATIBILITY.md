# VR Headset Compatibility Matrix

**Version**: 0.9.0
**Last Updated**: 2026-01-18

---

## Overview

Depth Surge 3D generates side-by-side stereoscopic 3D video compatible with most modern VR headsets. This document provides specifications and optimal settings for the top 10 most popular VR devices based on market share and Steam VR usage statistics.

---

## Top 10 VR Headsets by Popularity

Listed by market share and Steam VR usage (as of November 2025):

1. **Meta Quest 3** - 27.13% Steam VR market share
2. **Meta Quest 2** - Still widely used (46.02% in older data)
3. **Valve Index** - 14.36% Steam VR market share
4. **Oculus Rift S** - 13.10% (legacy device, declining)
5. **HTC Vive Pro 2** - Popular high-end PCVR
6. **Meta Quest 3S** - Entry-level Quest 3 (Oct 2024)
7. **PlayStation VR2** - Leading console VR platform
8. **HTC Vive Focus Vision** - Latest standalone (2024)
9. **Pico 4** - Quest competitor (Europe/Asia)
10. **HTC Vive XR Elite** - Mixed reality standalone (2023)

**Market Data Sources**:
- [Steam Hardware Survey](https://store.steampowered.com/hwsurvey/)
- [UploadVR SteamVR Usage Stats](https://www.uploadvr.com/steamvr-near-record-use-december-2024/)
- [DemandSage VR Statistics](https://www.demandsage.com/virtual-reality-statistics/)

---

## Headset Specifications

### 1. Meta Quest 3

**Official Specs**:
- **Resolution**: 2064 × 2208 per eye (4128 × 2208 combined)
- **IPD Range**: 53mm - 75mm (continuous scroll wheel adjustment)
- **Field of View**: 110° horizontal, 96° vertical
- **Refresh Rate**: 72Hz, 90Hz, 120Hz
- **Display Type**: Dual LCD with pancake lenses

**Recommended Output Settings**:
```
--vr-resolution 16x9-4k        # 3840x2160 side-by-side (1920x2160 per eye)
--vr-format side_by_side
--ipd 63                       # Average IPD (adjust per user)
--stereo-offset-pct 3.0        # Strong depth effect
```

**Sources**: [VRcompare](https://vr-compare.com/headset/metaquest3), [Wikipedia](https://en.wikipedia.org/wiki/Meta_Quest_3), [UploadVR](https://www.uploadvr.com/quest-3-specs/)

---

### 2. Meta Quest 2

**Official Specs**:
- **Resolution**: 1832 × 1920 per eye (3664 × 1920 combined)
- **IPD Range**: 58mm, 63mm, 68mm (3 fixed positions)
- **Field of View**: 96-97° (varies by lens separation)
- **Refresh Rate**: 72Hz, 90Hz, 120Hz
- **Display Type**: Single Fast-Switch LCD with Fresnel lenses

**Recommended Output Settings**:
```
--vr-resolution 16x9-1080p     # 1920x1080 side-by-side (960x1080 per eye)
--vr-format side_by_side
--ipd 63                       # Use 58/63/68 based on physical setting
--stereo-offset-pct 2.5
```

**Sources**: [VRcompare](https://vr-compare.com/headset/oculusquest2), [Meta Store](https://www.meta.com/quest/products/quest-2/tech-specs/)

---

### 3. Valve Index

**Official Specs**:
- **Resolution**: 1440 × 1600 per eye (2880 × 1600 combined)
- **IPD Range**: 58mm - 70mm (physical slider adjustment)
- **Field of View**: 108° horizontal, 104° vertical
- **Refresh Rate**: 80Hz, 90Hz, 120Hz, 144Hz
- **Display Type**: Dual RGB LCD

**Recommended Output Settings**:
```
--vr-resolution 16x9-1440p     # 2560x1440 side-by-side (1280x1440 per eye)
--vr-format side_by_side
--ipd 64                       # Average IPD
--stereo-offset-pct 2.8
```

**Notes**: Discontinued as of 2025, successor "Steam Frame" announced for early 2026.

**Sources**: [VRcompare](https://vr-compare.com/headset/valveindex), [Wikipedia](https://en.wikipedia.org/wiki/Valve_Index), [Valve Official](https://www.valvesoftware.com/en/index/headset)

---

### 4. Oculus Rift S (Legacy)

**Official Specs**:
- **Resolution**: 1280 × 1440 per eye (2560 × 1440 combined)
- **IPD Range**: Fixed 63.5mm (no adjustment)
- **Field of View**: ~90-100° (estimated)
- **Refresh Rate**: 80Hz
- **Display Type**: Single Fast-Switch LCD

**Recommended Output Settings**:
```
--vr-resolution 16x9-1440p     # 2560x1440 side-by-side
--vr-format side_by_side
--ipd 63.5                     # Fixed IPD
--stereo-offset-pct 2.5
```

**Notes**: Discontinued in 2021, replaced by Quest 2 with Link cable for PCVR.

---

### 5. HTC Vive Pro 2

**Official Specs**:
- **Resolution**: 2448 × 2448 per eye (4896 × 2448 combined)
- **IPD Range**: 57mm - 72mm (automatic adjustment)
- **Field of View**: 120° (wide FOV)
- **Refresh Rate**: 90Hz, 120Hz
- **Display Type**: Dual LCD with Fresnel lenses

**Recommended Output Settings**:
```
--vr-resolution 16x9-4k        # 3840x2160 side-by-side
--vr-format side_by_side
--ipd 63
--stereo-offset-pct 3.5        # Higher offset for wide FOV
```

**Sources**: [HTC Vive Official](https://www.vive.com/us/product/vive-pro2/specs/), [VRcompare](https://vr-compare.com/headset/htcvive)

---

### 6. Meta Quest 3S

**Official Specs**:
- **Resolution**: 1832 × 1920 per eye (same as Quest 2)
- **IPD Range**: 58mm, 63mm, 68mm (3 fixed positions)
- **Field of View**: 96° horizontal, 90° vertical
- **Refresh Rate**: 72Hz, 90Hz, 120Hz
- **Display Type**: Single Fast-Switch LCD with Fresnel lenses

**Recommended Output Settings**:
```
--vr-resolution 16x9-1080p     # 1920x1080 side-by-side
--vr-format side_by_side
--ipd 63                       # Use 58/63/68 based on physical setting
--stereo-offset-pct 2.5
```

**Notes**: Entry-level Quest 3 variant released Oct 2024, uses Quest 2 optics with Quest 3 chipset.

**Sources**: [VRcompare](https://vr-compare.com/headset/metaquest3s), [Wikipedia](https://en.wikipedia.org/wiki/Meta_Quest_3S), [Road to VR](https://www.roadtovr.com/quest-3s-quest-3-quest-2-specs-compared/)

---

### 7. PlayStation VR2 (PSVR2)

**Official Specs**:
- **Resolution**: 2000 × 2040 per eye (4000 × 2040 combined)
- **IPD Range**: Adjustable via lens slider (range not officially specified)
- **Field of View**: ~110°
- **Refresh Rate**: 90Hz, 120Hz
- **Display Type**: Dual OLED with Fresnel lenses

**Recommended Output Settings**:
```
--vr-resolution 16x9-4k        # 3840x2160 side-by-side
--vr-format side_by_side
--ipd 63
--stereo-offset-pct 3.0
```

**Notes**: PS5 exclusive, requires Media Player app to view side-by-side videos. Transfer files via USB or media server.

**Sources**: [PlayStation Official](https://www.playstation.com/en-us/ps-vr2/ps-vr2-tech-specs/), [VRcompare](https://vr-compare.com/headset/playstationvr2), [Wikipedia](https://en.wikipedia.org/wiki/PlayStation_VR2)

---

### 8. HTC Vive Focus Vision

**Official Specs**:
- **Resolution**: 2448 × 2448 per eye (4896 × 2448 combined)
- **IPD Range**: 57mm - 72mm (automatic adjustment)
- **Field of View**: 116-120°
- **Refresh Rate**: 90Hz, 120Hz
- **Display Type**: Dual LCD

**Recommended Output Settings**:
```
--vr-resolution 16x9-4k        # 3840x2160 side-by-side
--vr-format side_by_side
--ipd 63
--stereo-offset-pct 3.5        # Wide FOV benefits from stronger depth
```

**Notes**: Latest HTC standalone headset (2024), PC VR capable via DisplayPort.

**Sources**: [HTC Vive Official](https://www.vive.com/us/product/vive-focus-vision/specs/), [VRcompare](https://vr-compare.com/headset/htcvivefocusvision)

---

### 9. Pico 4

**Official Specs**:
- **Resolution**: 2160 × 2160 per eye (4320 × 2160 combined)
- **IPD Range**: 62mm - 72mm (motorized/seamless adjustment)
- **Field of View**: 105°
- **Refresh Rate**: 72Hz, 90Hz
- **Display Type**: Dual LCD with pancake lenses

**Recommended Output Settings**:
```
--vr-resolution 16x9-4k        # 3840x2160 side-by-side
--vr-format side_by_side
--ipd 67                       # Default center of range
--stereo-offset-pct 3.0
```

**Notes**: Not officially available in the US (Europe/Asia only). Quest 2/3 competitor.

**Sources**: [Pico Official](https://www.picoxr.com/global/products/pico4/specs), [VRcompare](https://vr-compare.com/headset/pico4), [UploadVR](https://www.uploadvr.com/pico-4-vs-quest-2-specs-features/)

---

### 10. HTC Vive XR Elite

**Official Specs**:
- **Resolution**: 1920 × 1920 per eye (3840 × 1920 combined)
- **IPD Range**: 54mm - 73mm (automatic adjustment)
- **Field of View**: Up to 110°
- **Refresh Rate**: 90Hz
- **Display Type**: Dual LCD with pancake lenses

**Recommended Output Settings**:
```
--vr-resolution 16x9-1080p     # 1920x1080 side-by-side
--vr-format side_by_side
--ipd 63.5
--stereo-offset-pct 3.0
```

**Notes**: Mixed reality standalone released 2023, lightweight design with detachable battery.

**Sources**: [HTC Vive Official](https://www.vive.com/us/product/vive-xr-elite/specs/), [Wikipedia](https://en.wikipedia.org/wiki/HTC_Vive_XR_Elite)

---

## Resolution Recommendations

### General Guidelines

**For best quality**, match or slightly exceed the per-eye resolution of your target headset:

| Headset Resolution (per eye) | Recommended Output | CLI Option |
|------------------------------|-------------------|------------|
| 1280 × 1440 (Rift S) | 1080p SBS | `--vr-resolution 16x9-1080p` |
| 1440 × 1600 (Index) | 1440p SBS | `--vr-resolution 16x9-1440p` |
| 1832 × 1920 (Quest 2/3S) | 1080p SBS | `--vr-resolution 16x9-1080p` |
| 1920 × 1920 (XR Elite) | 1080p SBS | `--vr-resolution 16x9-1080p` |
| 2000 × 2040 (PSVR2) | 4K SBS | `--vr-resolution 16x9-4k` |
| 2064 × 2208 (Quest 3) | 4K SBS | `--vr-resolution 16x9-4k` |
| 2160 × 2160 (Pico 4) | 4K SBS | `--vr-resolution 16x9-4k` |
| 2448 × 2448 (Vive Pro 2/Focus Vision) | 4K SBS | `--vr-resolution 16x9-4k` |

**Available Resolutions**:
- `16x9-1080p` - 1920×1080 side-by-side (960×1080 per eye)
- `16x9-1440p` - 2560×1440 side-by-side (1280×1440 per eye)
- `16x9-4k` - 3840×2160 side-by-side (1920×2160 per eye)

**Note**: Higher resolutions require significantly more processing time and VRAM. For source videos below 1080p, upscaling to 4K may not improve quality.

---

## IPD (Inter-Pupillary Distance) Settings

### What is IPD?

IPD is the distance between the centers of your pupils. Proper IPD adjustment is critical for:
- Comfortable viewing
- Correct depth perception
- Reducing eye strain

### Measuring Your IPD

1. **Use your VR headset's built-in IPD tool** (Quest 3, Focus Vision, etc.)
2. **Measure with a ruler**: Stand in front of a mirror, hold a ruler against your brow, measure pupil-to-pupil distance
3. **Ask an optometrist**: Your eyeglass prescription includes IPD
4. **Use a smartphone app**: Many free IPD measurement apps available

### Setting IPD in Depth Surge 3D

```bash
# Use your measured IPD value (in mm)
python depth_surge_3d.py input.mp4 --ipd 64

# Or via web UI: Configure in settings panel before processing
```

**Average IPD by demographics**:
- Adult males: 62-64mm
- Adult females: 58-62mm
- Children: 50-58mm
- General population average: 63mm

**Headset IPD ranges**:
- **Continuous adjustment**: Quest 3 (53-75mm), Index (58-70mm), Pico 4 (62-72mm)
- **Fixed positions**: Quest 2/3S (58/63/68mm), Rift S (63.5mm fixed)
- **Automatic**: Vive Pro 2, Focus Vision, XR Elite (software-controlled)

---

## Stereo Offset Settings

### What is Stereo Offset?

Stereo offset controls the perceived depth intensity by determining how much the left/right eye views are shifted based on depth information.

### Recommendations by Use Case

```bash
# Subtle depth (comfortable for extended viewing)
--stereo-offset-pct 2.0

# Moderate depth (balanced, recommended default)
--stereo-offset-pct 2.5

# Strong depth (dramatic 3D effect)
--stereo-offset-pct 3.0

# Very strong depth (wide FOV headsets like Vive Pro 2)
--stereo-offset-pct 3.5
```

**Guidelines**:
- Start with `2.5` for most headsets
- Increase for wide FOV headsets (>110°): Vive Pro 2, Focus Vision
- Decrease if experiencing eye strain or double vision
- Higher values = more dramatic 3D but can cause discomfort
- Lower values = more subtle depth but easier on eyes

---

## Viewing Your Videos in VR

### Meta Quest 2/3/3S

1. **Transfer video to headset**:
   - USB cable: Connect to PC, drag video to `Quest 3/Internal storage/Movies/`
   - Wireless: Use [SideQuest](https://sidequestvr.com/) file manager

2. **Playback apps**:
   - **Meta TV** (built-in): Plays side-by-side, limited controls
   - **Skybox VR Player** (recommended): Best controls, SMB/DLNA support
   - **DeoVR**: Good for 180°/360° content
   - **Pigasus VR Media Player**: Local files and network streaming

3. **Settings in Skybox**:
   - Video Format: "3D Side-by-Side"
   - Projection: "Flat" (for 2D-to-3D conversions)
   - FOV: Adjust to preference (default 90-100°)

### PlayStation VR2

1. **Transfer video to USB drive** (FAT32 or exFAT format)
2. **Connect to PS5 front USB port**
3. **Open Media Player app**
4. **Select video, press Options → Video Options → 3D Settings → Side-by-Side**

### PCVR Headsets (Index, Vive, Rift S)

1. **Use desktop VR video players**:
   - **Whirligig**: Powerful, many format options
   - **VR Video Player (Steam)**: Simple, effective
   - **DeoVR (Steam)**: Free, good quality

2. **Settings**:
   - Projection: Flat/Dome (for 2D-to-3D)
   - Stereo Mode: Side-by-Side
   - FOV: Match your headset (~100-110°)

### Standalone Headsets (Pico 4, Focus Vision, XR Elite)

1. **Transfer via USB or use DLNA/SMB server**
2. **Recommended apps**:
   - Pico Video (Pico 4 built-in)
   - Skybox VR
   - DeoVR

---

## Troubleshooting

### Double Vision / Eye Strain

**Cause**: IPD mismatch or excessive stereo offset

**Solutions**:
1. Verify IPD setting matches your measured IPD
2. Reduce `--stereo-offset-pct` (try 2.0 or 2.5)
3. Ensure headset IPD is properly adjusted
4. Take breaks every 20-30 minutes

### Flat Looking / No Depth

**Cause**: Insufficient stereo offset or poor depth map quality

**Solutions**:
1. Increase `--stereo-offset-pct` to 3.0-3.5
2. Use Video-Depth-Anything V2 model for better temporal consistency:
   ```bash
   --depth-model video-depth-anything-v2
   ```
3. Verify video player is set to "3D Side-by-Side" mode
4. Check that both eyes are displaying different views (cover one eye at a time)

### Blurry Video

**Cause**: Resolution too low for headset

**Solutions**:
1. Use higher VR resolution:
   ```bash
   --vr-resolution 16x9-4k  # For 2K+ headsets
   ```
2. Use high-quality source video (1080p or 4K source)
3. Enable AI upscaling:
   ```bash
   --upscale real-esrgan-4x
   ```

### Distorted Edges

**Cause**: VR player FOV mismatch or fisheye distortion issues

**Solutions**:
1. Disable fisheye distortion:
   ```bash
   --no-distortion
   ```
2. Adjust FOV in VR player to 90-100°
3. Use "Flat" projection mode (not dome/sphere)

---

## Performance Considerations

### Processing Time by Resolution

Estimated times for 1-minute 1080p source video @ 30fps on RTX 4070 Ti SUPER:

| Output Resolution | V3 Model | V2 Model | VRAM Usage |
|------------------|----------|----------|------------|
| 16x9-1080p | ~2 hours | ~3 hours | ~6GB |
| 16x9-1440p | ~2.5 hours | ~3.5 hours | ~8GB |
| 16x9-4k | ~3 hours | ~4 hours | ~10GB |

**Tips for faster processing**:
- Use V3 model (default) for 50% faster processing
- Process shorter clips (1-2 minutes)
- Use lower resolution for testing (`16x9-1080p`)
- Ensure GPU drivers are updated

---

## Contributing Headset Data

If you test Depth Surge 3D with a headset not listed here, please contribute your findings!

**What to share**:
1. Headset model and firmware version
2. Recommended VR resolution setting
3. Optimal IPD and stereo offset values
4. VR player app used
5. Any specific configuration tips

**How to contribute**:
- [Open an issue on GitHub](https://github.com/your-repo/depth-surge-3d/issues)
- Include example screenshots or video clips
- Note any compatibility issues

---

## Future Improvements

Planned enhancements for VR compatibility:

- [ ] Device-specific presets (auto-configure based on headset selection)
- [ ] Over-under format support for certain headsets
- [ ] FOV-matched distortion profiles
- [ ] Real-time preview in VR during processing
- [ ] IPD auto-detection from headset APIs
- [ ] Batch processing with per-headset profiles

See [TODO.md](TODO.md) for full roadmap.

---

## Additional Resources

### Official VR Headset Comparisons
- [VRcompare.com](https://vr-compare.com/) - Comprehensive spec database
- [HTC Vive Comparison Chart](https://www.vive.com/us/product/comparison/)
- [Meta Quest Comparison](https://www.meta.com/quest/compare/)

### VR Video Players
- [Skybox VR](https://skybox.xyz/)
- [DeoVR](https://deovr.com/)
- [Whirligig](http://www.whirligig.xyz/)

### VR Communities
- [r/virtualreality](https://reddit.com/r/virtualreality)
- [r/OculusQuest](https://reddit.com/r/OculusQuest)
- [SteamVR Forums](https://steamcommunity.com/app/250820/discussions/)

---

**Document Version**: 1.0
**Contributors**: Claude Sonnet 4.5, Community feedback welcome
