import { activate as activateVz } from "./virtual-zarr";
import { registerVirtualZarrSW } from "./sw-register";
import { mountNg } from "./ng-entry";
import { DEMO_SPEC, loadSpec, type ModelSpec } from "./model-spec";

const $ = <T extends HTMLElement>(id: string) => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as T;
};

const zarrInput = $<HTMLInputElement>("zarr-url");
const onnxInput = $<HTMLInputElement>("onnx-url");
const specInput = $<HTMLInputElement>("spec-url");
const activateBtn = $<HTMLButtonElement>("activate");
const status = $<HTMLSpanElement>("status");

function log(msg: string): void {
  status.textContent = msg;
}

activateBtn.addEventListener("click", async () => {
  activateBtn.disabled = true;
  try {
    log("registering service worker ...");
    await registerVirtualZarrSW();

    const specUrl = specInput.value.trim();
    let spec: ModelSpec = DEMO_SPEC;
    if (specUrl) {
      log(`loading model spec ${specUrl} ...`);
      spec = await loadSpec(specUrl);
    }

    log("opening zarr + loading model ...");
    const st = await activateVz({
      zarrUrl: zarrInput.value.trim(),
      modelUrl: onnxInput.value.trim(),
      spec,
    });

    // Output dim units = nm (multiscales axes use nanometer).
    const NM = 1e-9;
    mountNg({
      dimensions: { x: [NM, "m"], y: [NM, "m"], z: [NM, "m"] },
      position: [
        (st.outShape[2] * spec.outputVoxelSize[2]) / 2,
        (st.outShape[1] * spec.outputVoxelSize[1]) / 2,
        (st.outShape[0] * spec.outputVoxelSize[0]) / 2,
      ],
      crossSectionScale: NM * Math.max(...spec.outputVoxelSize),
      projectionScale:
        Math.max(
          st.outShape[0] * spec.outputVoxelSize[0],
          st.outShape[1] * spec.outputVoxelSize[1],
          st.outShape[2] * spec.outputVoxelSize[2],
        ) *
        NM *
        2,
      layers: [
        { type: "image", source: `zarr://${location.origin}/vz/`, name: "inference" },
      ],
      selectedLayer: { visible: true, layer: "inference" },
      layout: "4panel",
    });

    log(
      `active. out=${st.outShape.join("x")} (vox ${spec.outputVoxelSize.join(",")} nm) ` +
        `block=${spec.blockShape.join("x")} dtype=${spec.outputDtype} channels=${spec.outputChannels}`,
    );
  } catch (err) {
    const e = err as Error;
    log(`error: ${e.message}`);
    console.error(e);
  } finally {
    activateBtn.disabled = false;
  }
});
