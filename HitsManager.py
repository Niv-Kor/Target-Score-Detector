import Geometry2D as geo2D
import numpy as np

CANDIDATE = 0
VERIFIED = 1

candidate_hits = []
verified_hits = []

class Hit:
    def __init__(self, x, y, score, bullseyeRelation):
        '''
        {Number} x - x coordinate of the hit
        {Number} y - y coordinate of the hit
        {Number} score - The hit's score
        {Tuple} bullseysRelation - (
                                      {Number} current x coordinate of the bull'seye point,
                                      {Number} current y coordinate of the bull'seye point
                                   )
        '''

        self.point = (x, y)
        self.score = score
        self.reputation = 1
        self.bullseye_relation = bullseyeRelation
        
        # has this hit been checked during current iteration
        self.iter_mark = False
    
    def increase_rep(self):
        '''
        Increase the hit's reputation.
        '''

        self.reputation += 1
        
    def decrease_rep(self):
        '''
        Decrease the hit's reputation.
        '''

        self.reputation -= 1
        
    def isVerified(self, repScore):
        '''
        Parameters:
            {Number} repScore - The minimum reputation needed to verify a hit

        Returns:
            {Boolean} True if the hit is considered verified.
        '''

        return self.reputation >= repScore

def create_scoreboard(hits, scale, ringsAmount, innerDiam):
    '''
    Calculate the score of each detected hit.

    Parameters:
        {list} hits - [
                            {tuple} (
                                    {Number} x coordinates of the hit,
                                    {Number} y coordinates of the hit,
                                    {Number} The distance of the hit from the bull'seye
                                    )
                            ...
                       ]
        {tuple} scale - (
                            {Number} The percentage of the warped image's average horizontal edges' length
                                    out of the model's average horizontal edges' length,
                            {Number} The percentage of the warped image's average vertical edges' length
                                    out of the model's average vertical edges' length,
                            {Number} The size of the filmed target divided by the model target
                        )
    
    Returns:
        {list} [
                    {tuple} (
                                {tuple} (
                                        {Number} x coordinates of the hit,
                                        {Number} y coordinates of the hit
                                        ),
                                {Number} The hit's score according to the target's data
                            )
                    ...
                ]
    '''

    scoreboard = []
    
    for hit in hits:
        hit_dist = hit[2]
        scaled_diam = innerDiam * scale[2]
        score = 10 - int(hit_dist / scaled_diam)

        # clamp score between 10 and minimum available score
        if score < 10 - ringsAmount + 1:
            score = 0
        elif score > 10:
            score = 10
        
        hit_obj = Hit(int(hit[0]), int(hit[1]), score, hit[3])
        scoreboard.append(hit_obj)

    return scoreboard

def is_verified_hit(point, distanceTolerance):
    '''
    Parameters:
        {Tuple} point - (
                           {Number} x coordinate of the point,
                           {Number} y coordinate of the point
                        )
        {Number} distanceTolerance - Amount of pixels around the point that can be ignored
                                     in order to consider another point as the same one

    Returns:
        {Boolean} True if the point is of a verified hit.
    '''

    return type(get_hit(VERIFIED, point, distanceTolerance)) != type(None)

def is_candidate_hit(point, distanceTolerance):
    '''
    Parameters:
        {Tuple} point - (
                           {Number} x coordinate of the point,
                           {Number} y coordinate of the point
                        )
        {Number} distanceTolerance - Amount of pixels around the point that can be ignored
                                     in order to consider another point as the same one

    Returns:
        {Boolean} True if the point is of a known hit that's yet to be verified.
    '''
    
    return type(get_hit(CANDIDATE, point, distanceTolerance)) != type(None)

def get_hit(group, point, distanceTolerance):
    '''
    Parameters:
        {Number} group - The group to which the hit belongs
                            [HitsManager constant (VERIFIED, CANDIDATE)]
        {Tuple} point - (
                        {Number} x coordinate of the point,
                        {Number} y coordinate of the point
                        )
        {Number} distanceTolerance - Amount of pixels around the point that can be ignored
                                     in order to consider another point as the same one

    Returns:
        {HitsManager.Hit} The hit that's closest to the given point,
                            considering the tolarance distance around it.
                            If no hit is found, this function returns None.
    '''

    hits_list = get_hits(group)
    compatible_hits = []

    for hit in hits_list:
        if geo2D.euclidean_dist(point, hit.point) <= distanceTolerance:
            compatible_hits.append(hit)
            
    if len(compatible_hits) > 0:
        return compatible_hits[0]
    else:
        return None

def eliminate_verified_redundancy(distanceTolerance):
    '''
    Find duplicate verified hits and eliminate them.

    Parameters:
        {Number} distanceTolerance - Amount of pixels around a point that can be ignored
                                     in order to consider another point as the same one
    '''

    if len(verified_hits) <= 1:
        return
    
    # create a table of the distances between all hits
    table = np.ndarray((len(verified_hits),len(verified_hits)), np.float32)
    j_leap = 0
    for i in range(len(table)):
        for j in range(len(table[i])):
            col = j + j_leap
            if col >= len(table[i]):
                continue
            
            hit_i = verified_hits[i].point
            hit_j = verified_hits[col].point
            dist = geo2D.euclidean_dist(hit_i, hit_j)
            table[i][col] = dist
            
        j_leap += 1
    
    # find distances that are smaller than the threshold and eliminate the redundant hits
    j_leap = 0
    for i in range(len(table)):
        for j in range(len(table[i])):
            col = j + j_leap
            if col >= len(verified_hits):
                continue
            
            if i == col or i >= len(verified_hits):
                continue
            
            if table[i][col] < distanceTolerance:
                hit_i = verified_hits[i].point
                hit_j = verified_hits[col].point
                
                # check the distance from the bull'seye point
                bullseye_i = verified_hits[i].bullseye_relation
                bullseye_j = verified_hits[col].bullseye_relation
                bullseye_dist_i = geo2D.euclidean_dist(hit_i, bullseye_i)
                bullseye_dist_j = geo2D.euclidean_dist(hit_j, bullseye_j)

                if bullseye_dist_i < bullseye_dist_j:
                    verified_hits.remove(verified_hits[col])
                else:
                    verified_hits.remove(verified_hits[i])
        
        j_leap += 1

def sort_hit(hit, distanceTolerance, minVerifiedReputation):
    '''
    Sort a hit and place it in either of the lists.
    Increase the reputation of a hit that's already a candidate,
    or add a hit as a candidate if it's not already known.

    Parameters:
        {HitsManager.Hit} hit - The hit to sort
        {Number} distanceTolerance - Amount of pixels around a point that can be ignored
                                     in order to consider another point as the same one
        {Number} minVerifiedReputation - The minimum reputation needed to verify a hit
    '''

    candidate = get_hit(CANDIDATE, hit.point, distanceTolerance)

    # the hit is a known candidate
    if type(candidate) != type(None):
        candidate.increase_rep()
        candidate.iter_mark = True

        # candidate is now eligable for verification
        if candidate.isVerified(minVerifiedReputation):
            verified_hits.append(candidate)
            candidate_hits.remove(candidate)
            
            # find duplicate verified hits and eliminate them
            eliminate_verified_redundancy(distanceTolerance)

    # new candidate
    else:
        candidate_hits.append(hit)
        hit.iter_mark = True

def discharge_hits():
    '''
    Lower the reputation of hits that were not detected during the last iteration.
    Hits with reputation under 1 are disqualified and removed.
    '''

    for candidate in candidate_hits:
        # candidate is not present during the current iteration
        if not candidate.iter_mark:
            candidate.decrease_rep()
            
            # candidate disqualified
            if candidate.reputation <= 0:
                candidate_hits.remove(candidate)
                continue
        
        # get ready for the next iteration
        candidate.iter_mark = False

def shift_hits(bullseye):
    '''
    Shift all hits according to the new position of the bull'seye point in the target.

    Parameters:
        {Tuple} bullseye - (
                              {Number} current x coordinate of the bull'seye point in the target,
                              {Number} current y coordinate of the bull'seye point in the target
                           )
    '''

    all_hits = candidate_hits + verified_hits
    
    for h in all_hits:
        # find the correct translation amount
        x_dist = bullseye[0] - h.bullseye_relation[0]
        y_dist = bullseye[1] - h.bullseye_relation[1]
        new_x = int(round(h.point[0] + x_dist))
        new_y = int(round(h.point[1] + y_dist))
        
        # translate and update relation attribute
        h.bullseye_relation = bullseye
        h.point = (new_x,new_y)

def get_hits(group):
    '''
    Parameters:
        {Number} group - The group to which the hit belongs
                         [HitsManager constant (VERIFIED, CANDIDATE)]

    Returns:
        {List} The requested group of hits.
    '''

    switcher = {
        0: candidate_hits,
        1: verified_hits
    }

    return switcher.get(group, [])