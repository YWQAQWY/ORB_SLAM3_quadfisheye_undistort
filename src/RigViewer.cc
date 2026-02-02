/**
 * RigViewer: lightweight Pangolin viewer for cam0 trajectory.
 */

#include "RigViewer.h"
#include "Tracking.h"
#include "Atlas.h"
#include "MapPoint.h"

#include <pangolin/pangolin.h>

#include <chrono>
#include <thread>

namespace ORB_SLAM3
{

RigViewer::RigViewer(Tracking* pTracking, Atlas* pAtlas)
    : mpTracker(pTracking),
      mpAtlas(pAtlas),
      mbFinishRequested(false),
      mbFinished(true)
{
}

void RigViewer::Run()
{
    mbFinished = false;

    pangolin::CreateWindowAndBind("ORB-SLAM3: Rig Trajectory", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0.0, -10.0, -10.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0));

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while(!pangolin::ShouldQuit())
    {
        if(CheckFinish())
            break;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        if(mpTracker && mpTracker->mState == Tracking::OK && mpTracker->mCurrentFrame.HasPose())
        {
            Sophus::SE3f Tcw = mpTracker->mCurrentFrame.GetPose();
            Eigen::Vector3f Ow = Tcw.inverse().translation();

            std::lock_guard<std::mutex> lock(mMutexTrajectory);
            if(mTrajectory.empty() || (mTrajectory.back() - Ow).norm() > 1e-4f)
                mTrajectory.push_back(Ow);
        }

        if(mpAtlas)
        {
            std::vector<MapPoint*> map_points = mpAtlas->GetAllMapPoints();
            glPointSize(2.0f);
            glColor3f(0.6f, 0.6f, 0.9f);
            glBegin(GL_POINTS);
            for(auto *pMP : map_points)
            {
                if(!pMP || pMP->isBad())
                    continue;
                const Eigen::Vector3f pos = pMP->GetWorldPos();
                glVertex3f(pos.x(), pos.y(), pos.z());
            }
            glEnd();
        }

        {
            std::lock_guard<std::mutex> lock(mMutexTrajectory);
            if(!mTrajectory.empty())
            {
                glLineWidth(2.0f);
                glColor3f(0.1f, 0.9f, 0.1f);
                glBegin(GL_LINE_STRIP);
                for(const auto &p : mTrajectory)
                    glVertex3f(p.x(), p.y(), p.z());
                glEnd();

                glPointSize(6.0f);
                glColor3f(0.9f, 0.1f, 0.1f);
                glBegin(GL_POINTS);
                const auto &last = mTrajectory.back();
                glVertex3f(last.x(), last.y(), last.z());
                glEnd();
            }
        }

        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    SetFinish();
}

void RigViewer::RequestFinish()
{
    std::lock_guard<std::mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool RigViewer::isFinished()
{
    std::lock_guard<std::mutex> lock(mMutexFinish);
    return mbFinished;
}

bool RigViewer::CheckFinish()
{
    std::lock_guard<std::mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void RigViewer::SetFinish()
{
    std::lock_guard<std::mutex> lock(mMutexFinish);
    mbFinished = true;
}

}
