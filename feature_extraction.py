import hashlib
from androguard.core.apk import APK
from androguard.misc import AnalyzeAPK
import androguard.util

androguard.util.set_log("CRITICAL")


def calculate_file_hash(apk_path):
    sha256_hash = hashlib.sha256()
    with open(apk_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def extract_manifest(apk):
    permissions = apk.get_permissions()
    activities = apk.get_activities()
    services = apk.get_services()
    receivers = apk.get_receivers()

    return {
        "permissions": permissions,
        "activities": activities,
        "services": services,
        "receivers": receivers,
    }


def extract_intent_filters(apk):
    intent_filters = {"activities": {}, "services": {}, "receivers": {}}

    for activity in apk.get_activities():
        filters = apk.get_intent_filters("activity", activity)
        intent_filters["activities"][activity] = filters

    for service in apk.get_services():
        filters = apk.get_intent_filters("service", service)
        intent_filters["services"][service] = filters

    for receiver in apk.get_receivers():
        filters = apk.get_intent_filters("receiver", receiver)
        intent_filters["receivers"][receiver] = filters

    return intent_filters


def extract_apis(dx):
    api_calls = {}

    for method in dx.get_methods():
        if method.is_external():
            xrefs = method.get_xref_from()
            if xrefs:
                api_name = f"{method.class_name}->{method.name}{method.descriptor}"
                callers = []
                for _, caller_method, _ in xrefs:
                    callers.append(caller_method.full_name)

                api_calls[api_name] = callers

    return api_calls


def extract_opcodes(dx):
    opcodes = {}
    for method in dx.get_methods():
        if method.is_external():
            continue
        m = method.get_method()
        if m and m.get_code():
            for ins in m.get_instructions():
                op_name = ins.get_name()
                opcodes[op_name] = opcodes.get(op_name, 0) + 1
    return opcodes


def extract_features(data: dict) -> dict:
    feats = {}
    manifest = data.get("manifest", {})
    for perm in manifest.get("permissions", []):
        feats[f"perm_{perm}"] = 1.0
    for act in manifest.get("activities", []):
        feats[f"act_{act}"] = 1.0
    for srv in manifest.get("services", []):
        feats[f"srv_{srv}"] = 1.0
    for rec in manifest.get("receivers", []):
        feats[f"rec_{rec}"] = 1.0

    intent_filters = data.get("intentFilters", {})
    for comp_type, comp_dict in intent_filters.items():
        for comp_name, filters in comp_dict.items():
            for action in filters.get("action", []):
                feats[f"intent_action_{action}"] = 1.0
            for category in filters.get("category", []):
                feats[f"intent_category_{category}"] = 1.0

    for api_call in data.get("apiCalls", {}).keys():
        feats[f"api_{api_call}"] = 1.0

    for op_name, count in data.get("opcodes", {}).items():
        feats[f"opcode_{op_name}"] = float(count)

    return feats
