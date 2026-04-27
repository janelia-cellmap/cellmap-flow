import logging
import time

from flask import Blueprint, render_template

from cellmap_flow.norm.input_normalize import get_input_normalizers
from cellmap_flow.post.postprocessors import get_postprocessors_list
from cellmap_flow.models.model_merger import get_model_mergers_list
from cellmap_flow.globals import g

logger = logging.getLogger(__name__)

pipeline_builder_bp = Blueprint("pipeline_builder", __name__)


@pipeline_builder_bp.route("/pipeline-builder")
def pipeline_builder():
    """Render the drag-and-drop pipeline builder interface with current state from globals"""
    input_norms = get_input_normalizers()
    output_postprocessors = get_postprocessors_list()

    # Get available models from model catalog
    available_models = {}
    if hasattr(g, 'model_catalog') and g.model_catalog:
        for category, category_models in g.model_catalog.items():
            if isinstance(category_models, dict):
                for model_name, model_path in category_models.items():
                    full_name = f"{category}/{model_name}"
                    available_models[full_name] = {
                        'name': full_name,
                        'category': category,
                        'model_name': model_name,
                        'path': model_path
                    }

    # Also include any models from g.models_config if available
    if hasattr(g, 'models_config') and g.models_config:
        for model_config in g.models_config:
            model_dict = model_config.to_dict()
            model_name = model_config.name
            if model_name not in available_models:
                available_models[model_name] = model_dict

    logger.warning(f"\n{'='*80}")
    logger.warning(f"AVAILABLE MODELS DEBUG:")
    logger.warning(f"  Models from catalog: {[k for k in available_models.keys() if '/' in k]}")
    logger.warning(f"  Models from config: {[k for k in available_models.keys() if '/' not in k]}")
    logger.warning(f"  Total available models: {len(available_models)}")

    # Ensure all models have proper structure for UI
    models_with_config = {}
    for model_name, model_data in available_models.items():
        # Ensure the model has a 'name' field
        if isinstance(model_data, dict):
            models_with_config[model_name] = model_data
            if 'name' not in models_with_config[model_name]:
                models_with_config[model_name]['name'] = model_name
        else:
            models_with_config[model_name] = {'name': model_name}

    available_models = models_with_config
    logger.warning(f"  Final available_models count: {len(available_models)}")

    # Check if we have stored pipeline state from previous apply
    if hasattr(g, 'pipeline_normalizers') and len(g.pipeline_normalizers) > 0:
        # Use stored pipeline state (includes IDs, positions, params)
        current_normalizers = g.pipeline_normalizers
        current_postprocessors = g.pipeline_postprocessors
        current_models = g.pipeline_models
        # Enrich current_models with config from g.models_config if available
        if hasattr(g, 'models_config') and g.models_config:
            for model_dict in current_models:
                if 'config' not in model_dict:
                    # Strip _server suffix for matching
                    model_name = model_dict['name'].replace('_server', '')
                    for model_config in g.models_config:
                        config_name = getattr(model_config, 'name', '').replace('_server', '')
                        if config_name == model_name:
                            if hasattr(model_config, 'to_dict'):
                                model_dict['config'] = model_config.to_dict()
                            break
        current_inputs = g.pipeline_inputs
        current_outputs = g.pipeline_outputs
        current_edges = g.pipeline_edges
    else:
        # Fall back to converting from globals.input_norms and globals.postprocess
        current_normalizers = []
        for idx, norm in enumerate(g.input_norms):
            norm_dict = norm.to_dict() if hasattr(norm, 'to_dict') else {'name': str(norm)}
            norm_name = norm_dict.get('name', str(norm))
            # Extract params: all dict items except 'name'
            params = {k: v for k, v in norm_dict.items() if k != 'name'}
            current_normalizers.append({
                'id': f'norm-{idx}-{int(time.time()*1000)}',
                'name': norm_name,
                'params': params
            })

        # Current models (from jobs and models_config)
        current_models = []
        logger.warning(f"\n{'='*80}")
        logger.warning(f"Building current_models from g.jobs:")
        logger.warning(f"  g.jobs count: {len(g.jobs)}")
        logger.warning(f"  g.models_config exists: {hasattr(g, 'models_config')}")
        if hasattr(g, 'models_config'):
            logger.warning(f"  g.models_config count: {len(g.models_config) if g.models_config else 0}")
            logger.warning(f"  g.models_config type: {type(g.models_config)}")
            logger.warning(f"  g.models_config value: {g.models_config}")
            if g.models_config:
                logger.warning(f"  g.models_config names: {[getattr(mc, 'name', 'NO_NAME') for mc in g.models_config]}")
                for mc in g.models_config:
                    logger.warning(f"    Config object: {mc}, has to_dict: {hasattr(mc, 'to_dict')}")

        # If models_config is empty but we have jobs, try to get configs from model_catalog
        if (not hasattr(g, 'models_config') or not g.models_config) and hasattr(g, 'model_catalog'):
            logger.warning(f"  models_config is empty, checking model_catalog for configs...")
            # Check if available_models dict has configs
            if available_models:
                logger.warning(f"  available_models has {len(available_models)} entries with potential configs")

        for idx, job in enumerate(g.jobs):
            if hasattr(job, 'model_name'):
                logger.warning(f"\n  Processing job {idx}: model_name={job.model_name}")
                model_dict = {'id': f'model-{idx}-{int(time.time()*1000)}', 'name': job.model_name, 'params': {}}
                # Try to find the corresponding ModelConfig to get full configuration
                config_found = False

                # First try g.models_config
                if hasattr(g, 'models_config') and g.models_config:
                    # Strip _server suffix for matching
                    job_model_name = job.model_name.replace('_server', '')
                    for model_config in g.models_config:
                        model_config_name = getattr(model_config, 'name', None)
                        config_name_stripped = model_config_name.replace('_server', '') if model_config_name else None
                        logger.warning(f"    Checking model_config: {model_config_name} (stripped: {config_name_stripped}) vs job: {job.model_name} (stripped: {job_model_name})")
                        if config_name_stripped and config_name_stripped == job_model_name:
                            # Export the full model config using to_dict()
                            if hasattr(model_config, 'to_dict'):
                                model_dict['config'] = model_config.to_dict()
                                logger.warning(f"    ✓ Config attached from models_config: {model_dict['config']}")
                                config_found = True
                            break

                # Fallback: check available_models dict (which was enriched earlier)
                if not config_found and available_models:
                    job_model_name = job.model_name.replace('_server', '')
                    for model_name, model_data in available_models.items():
                        model_name_stripped = model_name.replace('_server', '')
                        logger.warning(f"    Checking available_models: {model_name} (stripped: {model_name_stripped}) vs job: {job.model_name} (stripped: {job_model_name})")
                        if model_name_stripped == job_model_name and isinstance(model_data, dict):
                            # Models from g.models_config store to_dict() directly
                            # (no nested 'config' key); use the dict itself as config
                            model_dict['config'] = model_data.get('config', model_data)
                            logger.warning(f"    ✓ Config attached from available_models: {model_dict['config']}")
                            config_found = True
                            break

                # Second fallback: check previously saved pipeline_model_configs
                if not config_found and hasattr(g, 'pipeline_model_configs'):
                    job_model_name = job.model_name.replace('_server', '')
                    for saved_name, saved_config in g.pipeline_model_configs.items():
                        saved_name_stripped = saved_name.replace('_server', '')
                        logger.warning(f"    Checking pipeline_model_configs: {saved_name} (stripped: {saved_name_stripped}) vs job: {job.model_name} (stripped: {job_model_name})")
                        if saved_name_stripped == job_model_name:
                            model_dict['config'] = saved_config
                            logger.warning(f"    ✓ Config attached from pipeline_model_configs: {model_dict['config']}")
                            config_found = True
                            break

                if not config_found:
                    logger.warning(f"    ✗ No matching config found for {job.model_name}")
                    logger.warning(f"       TIP: Import a YAML with full model configs to populate g.pipeline_model_configs")
                current_models.append(model_dict)
        logger.warning(f"{'='*80}\n")

        current_postprocessors = []
        for idx, post in enumerate(g.postprocess):
            post_dict = post.to_dict() if hasattr(post, 'to_dict') else {'name': str(post)}
            post_name = post_dict.get('name', str(post))
            # Extract params: all dict items except 'name'
            params = {k: v for k, v in post_dict.items() if k != 'name'}
            current_postprocessors.append({
                'id': f'post-{idx}-{int(time.time()*1000)}',
                'name': post_name,
                'params': params
            })

        current_inputs = []
        current_outputs = []
        current_edges = []

    # Get current dataset_path from globals
    dataset_path = getattr(g, 'dataset_path', None) or ''

    # Get available model mergers
    model_mergers = get_model_mergers_list()

    return render_template(
        "pipeline_builder_v2.html",
        input_normalizers=input_norms or {},
        available_models=available_models or {},
        output_postprocessors=output_postprocessors or {},
        model_mergers=model_mergers or {},
        current_normalizers=current_normalizers,
        current_models=current_models,
        current_postprocessors=current_postprocessors,
        current_inputs=current_inputs,
        current_outputs=current_outputs,
        current_edges=current_edges,
        dataset_path=dataset_path,
    )
