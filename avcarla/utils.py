def get_obj_type_from_actor(actor):
    h = 2 * actor.bounding_box.extent.z
    w = 2 * actor.bounding_box.extent.y
    l = 2 * actor.bounding_box.extent.x
    if "vehicle" in actor.type_id:
        n_wheels = actor.attributes["number_of_wheels"]
        if int(n_wheels) == 4:
            if h >= 2:
                obj_type = "truck"
            else:
                obj_type = "car"
        elif int(n_wheels) == 2:
            if any([mot in actor.type_id for mot in ["harley", "kawasaki", "yamaha"]]):
                obj_type = "motorcycle"
            else:
                obj_type = "bicycle"
        else:
            raise NotImplementedError(n_wheels)
    elif "walker" in actor.type_id:
        obj_type = "pedestrian"
        raise NotImplementedError(obj_type)
    else:
        raise NotImplementedError(actor.type_id)
    return obj_type
