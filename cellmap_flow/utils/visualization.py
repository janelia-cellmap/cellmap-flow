import random


def get_equivalences_shader(equivalences):
    ids = [item for sublist in equivalences for item in sublist]
    num_ids = len(ids)

    def generate_rgbas_string(equivalences):
        # Generate a list of vec4 strings
        vec4_list = []
        for current_equivalences in equivalences:
            # Generate three random numbers for r, g, b
            r = random.random()
            g = random.random()
            b = random.random()
            for _ in range(len(current_equivalences)):
                # Format the string for this vec4, you can adjust precision if needed
                vec4_list.append(f"vec4({r:.2f}, {g:.2f}, {b:.2f}, 1.0)")

        # Join the vec4 strings with commas and format into the final string
        vec4s_string = ",".join(vec4_list)
        final_string = f"vec4 rgba[NUM_IDS] = vec4[NUM_IDS]({vec4s_string});"
        return final_string

    def get_ids_string(ids):
        ids_string = ""
        for idx, id in enumerate(ids):
            low, high = get_low_high(id)
            ids_string += f"""ids[{idx}].value = uvec2({low}u, {high}u);"""
        return ids_string

    def get_low_high(value):
        # Mask for the lowest 32 bits
        low = value & 0xFFFFFFFF
        # The highest 32 bits are obtained by shifting right 32 bits
        high = value >> 32
        return low, high

    glsl_string = f"const int NUM_IDS = {num_ids};\n"
    glsl_string += "uint64_t ids[NUM_IDS];\n"
    glsl_string += generate_rgbas_string(equivalences)
    glsl_string += """

    // A simple pseudo-random function that returns a float between 0 and 1.
    float random(vec2 st) {
        return fract(sin(dot(st, vec2(12.9898, 78.233))) * 43758.5453123);
    }

    // Generate a random vec4 with the last component equal to 1.0.
    vec4 randomVec4(vec2 seed) {
        float r1 = random(seed + vec2(1.0, 0.0));
        float r2 = random(seed + vec2(0.0, 1.0));
        float r3 = random(seed + vec2(1.0, 1.0));
        return vec4(r1, r2, r3, 1.0);
    }

    // Helper function to compare two uint64_t values
    bool uint64Equals(uint64_t a, uint64_t b) {
        return a.value==b.value;
    }

    // Function that returns the corresponding rgba value for a given key
    vec4 getRGBA(uint64_t data) {
        // Loop through the ids array to find a match
        for (int i = 0; i < NUM_IDS; i++) {
            if (uint64Equals(data, ids[i])) {
                return rgba[i];
            }
        }
        // Return a default color (black with full alpha) if no match is found
        // Return a default color (black with full alpha) if no match is found
        if (data.value.x == 0u && data.value.y == 0u) {
            // Otherwise, return a vec4 of zeros.
            return vec4(0.0);
        
        } else {
            // Normalize the seed by dividing by 2**32 - 1 (i.e. 4294967295.0)
            return randomVec4(vec2(data.value.x, data.value.y) / 4294967295.0);
        }
    }
    void main() {
    """
    glsl_string += get_ids_string(ids)
    glsl_string += """
            emitRGBA(getRGBA(getDataValue()));
    } 
    """
    return glsl_string
