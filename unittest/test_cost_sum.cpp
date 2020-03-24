///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include "factory/cost.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_constructor() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // Test the initial size of the map
  BOOST_CHECK(model.get_costs().size() == 0);
}

void test_addCost() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // add an active cost
  boost::shared_ptr<crocoddyl::CostModelAbstract> rand_cost_1 = create_random_cost();
  model.addCost("random_cost_1", rand_cost_1, 1.);
  BOOST_CHECK(model.get_nr() == rand_cost_1->get_activation()->get_nr());

  // add an inactive cost
  boost::shared_ptr<crocoddyl::CostModelAbstract> rand_cost_2 = create_random_cost();
  model.addCost("random_cost_2", rand_cost_2, 1., false);
  BOOST_CHECK(model.get_nr() == rand_cost_1->get_activation()->get_nr());

  // change the random cost 2 status
  model.changeCostStatus("random_cost_2", true);
  BOOST_CHECK(model.get_nr() == rand_cost_1->get_activation()->get_nr() + rand_cost_2->get_activation()->get_nr());

  // change the random cost 1 status
  model.changeCostStatus("random_cost_1", false);
  BOOST_CHECK(model.get_nr() == rand_cost_2->get_activation()->get_nr());
}

void test_addCost_error_message() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // create an cost object
  boost::shared_ptr<crocoddyl::CostModelAbstract> rand_cost = create_random_cost();

  // add twice the same cost object to the container
  model.addCost("random_cost", rand_cost, 1.);

  // test error message when we add a duplicate cost
  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  model.addCost("random_cost", rand_cost, 1.);
  capture_ios.endCapture();
  std::stringstream expected_buffer;
  expected_buffer << "Warning: this cost item already existed, we cannot add it" << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());

  // test error message when we change the cost status of an inexistent cost
  capture_ios.beginCapture();
  model.changeCostStatus("no_exist_cost", true);
  capture_ios.endCapture();
  expected_buffer.clear();
  expected_buffer << "Warning: this cost item doesn't exist, we cannot change its status" << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());
}

void test_removeCost() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // add an active cost
  boost::shared_ptr<crocoddyl::CostModelAbstract> rand_cost = create_random_cost();
  model.addCost("random_cost", rand_cost, 1.);
  BOOST_CHECK(model.get_nr() == rand_cost->get_activation()->get_nr());

  // remove the cost
  model.removeCost("random_cost");
  BOOST_CHECK(model.get_nr() == 0);
}

void test_removeCost_error_message() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // remove a none existing cost form the container, we expect a cout message here
  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  model.removeCost("random_cost");
  capture_ios.endCapture();

  // Test that the error message is sent.
  std::stringstream expected_buffer;
  expected_buffer << "Warning: this cost item doesn't exist, we cannot remove it" << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());
}

void test_get_costs() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some contact objects
  for (unsigned i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_cost_" << i;
    model.addCost(os.str(), create_random_cost(), 1.);
  }

  // get the contacts
  const crocoddyl::CostModelSum::CostModelContainer& costs = model.get_costs();

  // test
  crocoddyl::CostModelSum::CostModelContainer::const_iterator it_m, end_m;
  unsigned i;
  for (i = 0, it_m = costs.begin(), end_m = costs.end(); it_m != end_m; ++it_m, ++i) {
    std::ostringstream os;
    os << "random_cost_" << i;
    BOOST_CHECK(it_m->first == os.str());
  }
}

void test_get_nr() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some contact objects
  for (unsigned i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_cost_" << i;
    model.addCost(os.str(), create_random_cost(), 1.);
  }

  // compute ni
  std::size_t nr = 0;
  crocoddyl::CostModelSum::CostModelContainer::const_iterator it_m, end_m;
  for (it_m = model.get_costs().begin(), end_m = model.get_costs().end(); it_m != end_m; ++it_m) {
    nr += it_m->second->cost->get_activation()->get_nr();
  }

  BOOST_CHECK(nr == model.get_nr());
}

//----------------------------------------------------------------------------//

void register_unit_tests() {
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_constructor)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_addCost)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_addCost_error_message)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_removeCost)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_removeCost_error_message)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_get_costs)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_get_nr)));
}

bool init_function() {
  register_unit_tests();
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }