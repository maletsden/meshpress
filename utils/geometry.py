def triangle_list_to_generalized_strip(triangle_list):
    """
    Convert a triangle list to a generalized triangle strip with side information.
    Separately returns the generalized triangle strip vertices and side bits.

    Parameters:
    triangle_list (list): A list of triangles, where each triangle is represented by three vertices
                          [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)].

    Returns:
    tuple: A tuple containing:
           - strip_vertices (list): A list of vertices representing the triangle strip.
           - side_bits (list): A list of side bits (0 for left, 1 for right) or None for degenerate triangles.
    """
    if len(triangle_list) < 2:
        return [v for triangle in triangle_list for v in triangle], [None] * len(triangle_list)

    strip_vertices = []
    side_bits = []
    used = [False] * len(triangle_list)

    # Start with the first triangle
    strip_vertices.extend(triangle_list[0])
    side_bits.extend([None, None, None])  # Initial triangle has no side bits
    used[0] = True
    current_triangle = triangle_list[0]

    while not all(used):
        found_shared = False

        # Search for the next triangle that shares an edge with the current triangle
        for i in range(len(triangle_list)):
            if used[i]:
                continue

            next_triangle = triangle_list[i]

            is_left_triangle = [
                (next_triangle[0] == current_triangle[0] and next_triangle[1] == current_triangle[2]),
                (next_triangle[0] == current_triangle[2] and next_triangle[2] == current_triangle[0]),
                (next_triangle[1] == current_triangle[0] and next_triangle[2] == current_triangle[2])
            ]

            is_right_triangle = [
                (next_triangle[0] == current_triangle[2] and next_triangle[1] == current_triangle[1]),
                (next_triangle[0] == current_triangle[1] and next_triangle[2] == current_triangle[2]),
                (next_triangle[1] == current_triangle[2] and next_triangle[2] == current_triangle[1])
            ]

            if not any(is_left_triangle) and not any(is_right_triangle):
                continue

            next_vertex = (
                    is_left_triangle[0] * next_triangle[2] +
                    is_left_triangle[1] * next_triangle[1] +
                    is_left_triangle[2] * next_triangle[0] +
                    is_right_triangle[0] * next_triangle[2] +
                    is_right_triangle[1] * next_triangle[1] +
                    is_right_triangle[2] * next_triangle[0]
            )
            # print(f"{next_vertex = }")
            strip_vertices.append(next_vertex)
            side_bit = 0 if any(is_left_triangle) else 1
            side_bits.append(side_bit)
            used[i] = True
            found_shared = True

            # current_triangle = next_triangle
            current_triangle = (
                [current_triangle[0], current_triangle[2], next_vertex]
                if side_bit == 0 else [current_triangle[2], current_triangle[1], next_vertex]
            )
            break

        if not found_shared:
            # No shared edge found; insert degenerate triangles to connect to the next unused triangle
            next_triangle = None
            for i in range(len(triangle_list)):
                if not used[i]:
                    next_triangle = triangle_list[i]
                    used[i] = True
                    break

            # We don't need that
            strip_vertices.extend(
                [strip_vertices[-1], next_triangle[0], next_triangle[0], next_triangle[1], next_triangle[2]])
            side_bits.extend([0, 1, 1, 0, 0])

            # strip_vertices.extend([strip_vertices[-1], next_triangle[0], next_triangle[1], next_triangle[2]])
            # side_bits.append(0)

            current_triangle = next_triangle

    return strip_vertices, side_bits


def generalized_strip_to_triangle_list(strip_vertices, side_bits):
    """
    Convert a generalized triangle strip back to a triangle list.

    Parameters:
    strip_vertices (list): A list of vertices representing the generalized triangle strip.
    side_bits (list): A list of side bits (0 for left, 1 for right, or None for initial vertices and degenerate triangles).

    Returns:
    list: A list of triangles, where each triangle is represented by three vertices [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)].
    """
    if len(strip_vertices) < 3:
        return []

    triangle_list = []
    # Start with the first triangle formed by the first three vertices
    current_triangle = [strip_vertices[0], strip_vertices[1], strip_vertices[2]]
    triangle_list.append(tuple(current_triangle))

    # Iterate through the rest of the vertices and side_bits to reconstruct triangles
    vertex_idx = 3
    bit_idx = 3
    while vertex_idx < len(strip_vertices):
        # for i in range(3, len(strip_vertices)):
        new_vertex = strip_vertices[vertex_idx]
        side_bit = side_bits[bit_idx]

        # Check the side_bit to determine the next triangle vertices
        if side_bit == 0:  # Use left side
            current_triangle = [current_triangle[0], current_triangle[2], new_vertex]
        else:
            current_triangle = [current_triangle[2], current_triangle[1], new_vertex]

        vertex_idx += 1
        bit_idx += 1
        if current_triangle[0] == current_triangle[1] or current_triangle[0] == current_triangle[2] or current_triangle[
            1] == current_triangle[2]:
            # Skip degenerate triangles

            # current_triangle = [strip_vertices[vertex_idx + 1], strip_vertices[vertex_idx + 2],
            #                     strip_vertices[vertex_idx + 3]]
            # # triangle_list.append(tuple(current_triangle))
            # vertex_idx += 3
            # # bit_idx += 1

            continue
        triangle_list.append(tuple(current_triangle))

    return triangle_list


def triangle_list_to_strip(triangle_list):
    """
    Convert a triangle list to a triangle strip with degenerate triangles if needed.

    Parameters:
    triangle_list (list): A list of triangles, where each triangle is represented by three vertices [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)].

    Returns:
    list: A list of vertices representing the triangle strip, including degenerate triangles if needed.
    """
    if len(triangle_list) < 2:
        return [v for triangle in triangle_list for v in triangle]

    strip = []
    used = [False] * len(triangle_list)

    # Start with the first triangle
    strip.extend(triangle_list[0])
    used[0] = True

    current_triangle = triangle_list[0]

    while not all(used):
        found_shared = False

        # Search for the next triangle that shares an edge with the current triangle
        for i in range(len(triangle_list)):
            if used[i]:
                continue

            t2 = triangle_list[i]

            shared_vertices = [v for v in t2 if v in [strip[-1], strip[-2]]]

            if len(shared_vertices) == 2:
                # Triangles share an edge, add the new vertex while preserving order
                new_vertex = [v for v in t2 if v not in shared_vertices][0]
                # if current_triangle.index(shared_vertices[0]) < current_triangle.index(shared_vertices[1]):
                #     strip.append(new_vertex)
                # else:
                strip.append(new_vertex)
                # current_triangle = t2
                used[i] = True
                found_shared = True
                break

        if not found_shared:
            # No shared edge found, add degenerate triangles to connect
            # strip.append(current_triangle[2])  # Repeat the last vertex of the current triangle
            strip.append(strip[-1])  # Repeat the last vertex of the current triangle
            for i in range(len(triangle_list)):
                if not used[i]:
                    strip.append(triangle_list[i][0])  # Repeat the first vertex of the next triangle

                    if len(strip) % 2 == 0:
                        strip.extend([triangle_list[i][0], triangle_list[i][1], triangle_list[i][2]])

                    else:
                        strip.extend([triangle_list[i][0], triangle_list[i][2], triangle_list[i][1]])

                    # strip.extend(triangle_list[i])     # Add the new triangle
                    # current_triangle = triangle_list[i]
                    used[i] = True
                    break

    return strip


def triangle_strip_to_list(triangle_strip):
    """
    Convert a triangle strip to a triangle list.

    Parameters:
    triangle_strip (list): A list of vertices representing the triangle strip.

    Returns:
    list: A list of triangles, where each triangle is represented by three vertices.
    """
    if len(triangle_strip) < 3:
        return []

    triangle_list = []

    # Iterate through the strip to form triangles
    for i in range(2, len(triangle_strip)):
        if triangle_strip[i] == triangle_strip[i - 1] or triangle_strip[i] == triangle_strip[i - 2] or triangle_strip[
            i - 1] == triangle_strip[i - 2]:
            # Skip degenerate triangles
            continue
        if i % 2 == 0:
            triangle = (triangle_strip[i - 2], triangle_strip[i - 1], triangle_strip[i])
        else:
            triangle = (triangle_strip[i - 2], triangle_strip[i], triangle_strip[i - 1])
        triangle_list.append(triangle)

    return triangle_list


def reorder_vertices(vertices, triangle_strip, indices_remap):
    indices_remapping_inverse = {new: old for old, new in indices_remap.items()}
    reordered_vertices = [vertices[indices_remapping_inverse[i]] for i in range(len(vertices))]
    reordered_triangle_strip = [indices_remap[v] for v in triangle_strip]

    return reordered_vertices, reordered_triangle_strip


def check_reconstructed_triangles(triangles, reconstructed_triangles):
    found_triangles = [False] * len(triangles)

    for i, triangle in enumerate(triangles):
        if (triangle[0], triangle[1], triangle[2]) in reconstructed_triangles or (
                triangle[2], triangle[0], triangle[1]) in reconstructed_triangles or (
                triangle[1], triangle[2], triangle[0]) in reconstructed_triangles:
            found_triangles[i] = True
            continue
        print(f"Triangle not found: {triangle} ({i})")

    return all(found_triangles)
