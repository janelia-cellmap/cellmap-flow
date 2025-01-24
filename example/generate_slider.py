#%%
def generate_script(num_channels=2):
    """
    Generates a script with the following pattern:
    - For each channel i in [0..num_channels-1]:
      #uicontrol bool show_ch{i} checkbox()
      #uicontrol invlerp ch{i}(channel={i}, range=[0, 255], window=[0, 255]);
      #uicontrol vec3 color{i} color(default="red")
    - main() function that emits the sum of each channel's color * channel value * bool show.

    :param num_channels: The number of channels to generate UI controls and code for.
    :return: A string containing the generated script.
    """
    lines = []
    
    # Generate #uicontrol lines
    # for i in range(num_channels):
    #     lines.append(f"#uicontrol bool show_ch{i} checkbox()")
    for i in range(num_channels):
        lines.append(f"#uicontrol invlerp ch{i}(channel={i}, range=[0, 255], window=[0, 255]);")
    for i in range(num_channels):
        lines.append(f"#uicontrol vec3 color{i} color(default=\"red\")")
    
    # Build the emit line
    # Example piece for channel i: "color{i} * ch{i}() * float(show_ch{i})"
    # emit_parts = [f"color{i} * ch{i}() * float(show_ch{i})/float({num_channels})" for i in range(num_channels)]
    emit_parts = [f"color{i} * ch{i}()/float({num_channels})" for i in range(num_channels)]
    
    # Wrap up in the main
    lines.append("")
    lines.append("void main() {")
    lines.append(f"  emitRGB({' + '.join(emit_parts)});")
    lines.append("}")
    
    # Return the generated script as a single string
    return "\n".join(lines)


# Example usage:
if __name__ == "__main__":
    # Generate a script for 2 channels (default)
    script_2_channels = generate_script(8)
    print("Generated script for 8 channels:")
    print(script_2_channels)
    
    # print("\n" + "-"*50 + "\n")
    
    # # Generate a script for, say, 4 channels
    # script_4_channels = generate_script(4)
    # print("Generated script for 4 channels:")
    # print(script_4_channels)

# %%
